"""AERCA 后端：FastAPI + WebSocket。

启动：  uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 项目根加入 sys.path，复用 datasets/args/models/main.py
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.dataset_registry import DATASET_REGISTRY, get_default_options, list_datasets
from backend.session_store import Session, store

logger = logging.getLogger("aerca.backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ============================================================
# Pydantic Schemas
# ============================================================
class CreateSessionRequest(BaseModel):
    dataset_name: str
    adtype: Optional[str] = None
    fault_id: Optional[str] = None
    preprocessing_data: int = 1  # 1=生成新数据，0=加载已有
    seed: int = 42
    # 可选覆盖项
    training_size: Optional[int] = None
    testing_size: Optional[int] = None
    T: Optional[int] = None  # 序列长度
    device: Optional[str] = None  # 'cuda' / 'cpu' / 'mps'，留空则自动检测


class CreateSessionResponse(BaseModel):
    session_id: str
    dataset_name: str
    options_summary: Dict[str, Any]
    num_vars: int
    training_size: int
    testing_size: int
    T: int
    has_causal_struct: bool
    use_slice: bool
    supports_adtype: bool
    fault_id: Optional[str] = None
    fault_name: Optional[str] = None
    fault_category: Optional[str] = None
    fault_pattern: Optional[str] = None


class RunModelRequest(BaseModel):
    epochs: int = Field(50, ge=1, le=10000)
    lr: float = Field(1e-3, gt=0)
    training_aerca: bool = True


class StreamInjectRequest(BaseModel):
    fault_id: str = "equipment_spike"


# ============================================================
# 应用
# ============================================================
app = FastAPI(title="AERCA Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def get_datasets():
    return {"datasets": list_datasets()}


@app.get("/api/datasets/{name}/defaults")
def get_dataset_defaults(name: str):
    if name not in DATASET_REGISTRY:
        raise HTTPException(404, f"Unknown dataset: {name}")
    opts = get_default_options(name)
    # 仅返回可序列化的标量项
    safe = {k: v for k, v in opts.items() if isinstance(v, (int, float, str, bool, type(None)))}
    return {"defaults": safe}


# ============================================================
# 创建会话 → 生成或加载数据
# ============================================================
def _resolve_fault_for_session(dataset_name: str, fault_id: Optional[str]) -> Optional[Dict[str, Any]]:
    dataset_faults = globals().get("STREAM_FAULTS_BY_DATASET", {}).get(dataset_name, {})
    if not dataset_faults:
        return None
    resolved_id = fault_id or next(iter(dataset_faults))
    if resolved_id not in dataset_faults:
        raise HTTPException(400, f"Unknown fault_id for {dataset_name}: {resolved_id}")
    return dataset_faults[resolved_id]


@app.post("/api/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    name = req.dataset_name
    if name not in DATASET_REGISTRY:
        raise HTTPException(400, f"Unknown dataset: {name}")
    info = DATASET_REGISTRY[name]

    options = get_default_options(name)

    # 自动检测可用设备：优先用户指定 → cuda → mps → cpu
    if req.device:
        device = req.device
    else:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    options.update({
        "dataset_name": name,
        "preprocessing_data": req.preprocessing_data,
        "seed": req.seed,
        "training_aerca": True,
        "device": device,
    })
    if req.training_size is not None:
        options["training_size"] = req.training_size
    if req.testing_size is not None:
        options["testing_size"] = req.testing_size
    if req.T is not None:
        options["T"] = req.T

    fault_spec = _resolve_fault_for_session(name, req.fault_id)
    if fault_spec is not None:
        options["fault_id"] = fault_spec["id"]
        options["fault_name"] = fault_spec["name"]
        options["fault_category"] = fault_spec["category"]
        options["fault_pattern"] = fault_spec["pattern"]
        options["fault_affected_vars"] = ",".join(str(v) for v in fault_spec["affected_vars"])

    # adtype 处理：新的故障场景优先决定异常形态，旧 adtype 仅作为兼容兜底。
    if info["supports_adtype"] and fault_spec is not None:
        options["adtype"] = fault_spec["pattern"]
    elif info["supports_adtype"] and req.adtype:
        options["adtype"] = req.adtype
    elif info["adtypes"]:
        options["adtype"] = info["adtypes"][0]

    DataClass = info["class"]
    try:
        data_class = DataClass(options)
        if req.preprocessing_data == 1:
            data_class.generate_example()
            data_class.save_data()
        else:
            data_class.load_data()
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to create dataset")
        raise HTTPException(500, f"Dataset error: {e}") from e

    session = store.create(name, options, data_class)

    has_causal = "causal_struct" in data_class.data_dict
    sample = data_class.data_dict.get("x_n_list")
    num_vars = int(sample.shape[2]) if sample is not None and len(sample) > 0 else int(options.get("num_vars", 0))

    return CreateSessionResponse(
        session_id=session.session_id,
        dataset_name=name,
        options_summary={
            k: v for k, v in options.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        },
        num_vars=num_vars,
        training_size=int(options.get("training_size", 0)),
        testing_size=int(options.get("testing_size", 0)),
        T=int(options.get("T", sample.shape[1] if sample is not None and len(sample) > 0 else 0)),
        has_causal_struct=has_causal,
        use_slice=info["use_slice"],
        supports_adtype=info["supports_adtype"],
        fault_id=options.get("fault_id"),
        fault_name=options.get("fault_name"),
        fault_category=options.get("fault_category"),
        fault_pattern=options.get("fault_pattern"),
    )


def _require_session(session_id: str) -> Session:
    s = store.get(session_id)
    if s is None:
        raise HTTPException(404, f"Session not found: {session_id}")
    return s


# ============================================================
# 数据查询 API
# ============================================================
@app.get("/api/sessions/{sid}/info")
def session_info(sid: str):
    s = _require_session(sid)
    info = DATASET_REGISTRY[s.dataset_name]
    sample = s.data_class.data_dict.get("x_n_list")
    n_total = int(sample.shape[0]) if sample is not None and len(sample) > 0 else 0
    if info["use_slice"]:
        training_size = int(s.options.get("training_size", 0))
        testing_size = max(0, n_total - training_size)
    else:
        training_size = 0
        testing_size = n_total

    return {
        "session_id": s.session_id,
        "dataset_name": s.dataset_name,
        "run_status": s.run_status,
        "n_total_samples": n_total,
        "training_size": training_size,
        "testing_size": testing_size,
        "use_slice": info["use_slice"],
        "supports_adtype": info["supports_adtype"],
        "num_vars": int(sample.shape[2]) if sample is not None and len(sample) > 0 else 0,
        "T": int(sample.shape[1]) if sample is not None and len(sample) > 0 else 0,
        "has_causal_struct": "causal_struct" in s.data_class.data_dict,
        "fault_id": s.options.get("fault_id"),
        "fault_name": s.options.get("fault_name"),
        "fault_category": s.options.get("fault_category"),
        "fault_pattern": s.options.get("fault_pattern"),
    }


@app.get("/api/sessions/{sid}/sample/{idx}")
def get_sample(sid: str, idx: int, source: str = "auto"):
    """获取单个样本数据。

    source:
      - auto:   未运行模型时取 x_ab_list[idx], x_n_list[idx]；
                运行后取测试集 test_x_ab[idx]，正常对应 x_n_list[idx + training_size]
      - raw:    强制使用原始 x_n_list/x_ab_list/label_list 直接索引
    """
    s = _require_session(sid)
    d = s.data_class.data_dict
    info = DATASET_REGISTRY[s.dataset_name]

    has_run = s.run_status == "done" and s.results is not None
    use_test = (source == "auto") and has_run and info["use_slice"]

    if use_test:
        results = s.results
        test_x_ab = results["test_x_ab"]
        test_label = results["test_label"]
        if idx < 0 or idx >= len(test_x_ab):
            raise HTTPException(400, "idx out of range (test set)")
        x_ab = test_x_ab[idx]
        label = test_label[idx]
        offset = int(s.options.get("training_size", 0))
        x_n_full = d.get("x_n_list")
        if x_n_full is not None and (idx + offset) < len(x_n_full):
            x_n = x_n_full[idx + offset]
        else:
            x_n = np.zeros_like(x_ab)
    else:
        x_n_full = d.get("x_n_list")
        x_ab_full = d.get("x_ab_list")
        label_full = d.get("label_list")
        if x_ab_full is None:
            raise HTTPException(404, "No x_ab_list available")
        if idx < 0 or idx >= len(x_ab_full):
            raise HTTPException(400, "idx out of range")
        x_ab = x_ab_full[idx]
        x_n = x_n_full[idx] if x_n_full is not None and idx < len(x_n_full) else np.zeros_like(x_ab)
        label = label_full[idx] if label_full is not None and idx < len(label_full) else np.zeros_like(x_ab)

    return {
        "idx": idx,
        "T": int(x_ab.shape[0]),
        "num_vars": int(x_ab.shape[1]),
        "x_n": x_n.tolist(),
        "x_ab": x_ab.tolist(),
        "label": np.asarray(label).astype(int).tolist(),
        "from_test_set": bool(use_test),
    }


@app.get("/api/sessions/{sid}/causal")
def get_causal(sid: str):
    s = _require_session(sid)
    cs = s.data_class.data_dict.get("causal_struct")
    if cs is None:
        raise HTTPException(404, "No causal_struct in this dataset")
    return {"matrix": np.asarray(cs).tolist()}


# ============================================================
# 模型运行（异步任务 + WebSocket 进度）
# ============================================================
def _to_serializable(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _make_progress_callback(session: Session, loop: asyncio.AbstractEventLoop):
    def _cb(info: Dict[str, Any]):
        safe_info = {k: _to_serializable(v) for k, v in info.items()}
        session.push_progress(safe_info, loop)

    def _should_stop() -> bool:
        return bool(session.training_stop_requested)

    _cb.should_stop = _should_stop  # type: ignore[attr-defined]
    return _cb


def _run_model_blocking(session: Session, req: RunModelRequest, loop: asyncio.AbstractEventLoop):
    """在线程池中阻塞执行模型训练 + 测试。"""
    try:
        session.run_status = "running"
        session.run_error = None
        session.results = None
        session.training_stop_requested = False
        session.progress_log.clear()
        cb = _make_progress_callback(session, loop)
        cb({"phase": "starting", "message": "Initializing model..."})

        # 通过 sys.argv hack 传给 main()
        from main import main as run_main

        argv = ["main.py", "--dataset_name", session.dataset_name]
        info = DATASET_REGISTRY[session.dataset_name]
        if info["supports_adtype"] and "adtype" in session.options:
            argv += ["--adtype", str(session.options["adtype"])]
        if "fault_id" in session.options:
            argv += ["--fault_id", str(session.options["fault_id"])]
        argv += ["--epochs", str(req.epochs), "--lr", str(req.lr)]
        argv += ["--training_aerca", "1" if req.training_aerca else "0"]
        argv += ["--preprocessing_data", "0"]  # 已用 data_class 注入，跳过生成
        # 透传 device 与样本规模，保持与 session 创建时一致
        for k in ("device", "training_size", "testing_size", "T", "seed"):
            if k in session.options:
                argv += [f"--{k}", str(session.options[k])]

        results = run_main(argv, progress_callback=cb, data_class=session.data_class)

        # 把 results 缓存（不可序列化字段保留为 numpy，仅在 API 输出时 .tolist）
        session.results = results
        session.run_status = "done"
        session.training_stop_requested = False
        cb({"phase": "done", "message": "All stages complete."})
    except InterruptedError as e:
        logger.info("Model run stopped: %s", e)
        session.run_status = "stopped"
        session.run_error = None
        session.results = None
        session.training_stop_requested = False
        try:
            session.push_progress(
                {"phase": "stopped", "message": str(e) or "Training stopped by user."},
                loop,
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        logger.exception("Model run failed")
        session.run_status = "failed"
        session.run_error = f"{type(e).__name__}: {e}"
        session.training_stop_requested = False
        try:
            session.push_progress(
                {"phase": "error", "message": session.run_error, "trace": traceback.format_exc()},
                loop,
            )
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/sessions/{sid}/run")
async def run_model(sid: str, req: RunModelRequest):
    s = _require_session(sid)
    if s.run_status == "running":
        raise HTTPException(409, "A run is already in progress")
    loop = asyncio.get_running_loop()
    # 用线程池跑阻塞任务
    loop.run_in_executor(None, _run_model_blocking, s, req, loop)
    return {"status": "started", "session_id": sid}


@app.post("/api/sessions/{sid}/run/stop")
def stop_model_run(sid: str):
    s = _require_session(sid)
    if s.run_status != "running":
        return {"status": "not_running", "run_status": s.run_status}
    s.training_stop_requested = True
    return {"status": "stopping", "session_id": sid}


@app.get("/api/sessions/{sid}/status")
def run_status(sid: str):
    s = _require_session(sid)
    return {
        "session_id": sid,
        "run_status": s.run_status,
        "run_error": s.run_error,
        "progress_count": len(s.progress_log),
        "last_progress": s.progress_log[-1] if s.progress_log else None,
    }


@app.get("/api/sessions/{sid}/progress")
def run_progress_log(sid: str):
    s = _require_session(sid)
    return {"log": s.progress_log}


# ============================================================
# WebSocket：推送实时训练进度
# ============================================================
@app.websocket("/api/sessions/{sid}/ws")
async def ws_progress(websocket: WebSocket, sid: str):
    await websocket.accept()
    s = store.get(sid)
    if s is None:
        await websocket.send_json({"phase": "error", "message": f"Session not found: {sid}"})
        await websocket.close()
        return

    queue = s.add_subscriber()
    try:
        # 立即发送当前状态
        await websocket.send_json({"phase": "hello", "run_status": s.run_status})
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(msg)
                if msg.get("phase") in ("done", "error"):
                    # 不立即关闭：让客户端拿到信号后自行关闭
                    pass
            except asyncio.TimeoutError:
                # 心跳，保活
                await websocket.send_json({"phase": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("WebSocket error")
    finally:
        s.remove_subscriber(queue)


# ============================================================
# 结果查询
# ============================================================
@app.get("/api/sessions/{sid}/results")
def get_results(sid: str):
    s = _require_session(sid)
    if s.run_status != "done" or s.results is None:
        raise HTTPException(409, f"No results yet (status={s.run_status})")
    r = s.results

    out: Dict[str, Any] = {
        "training_size": int(r.get("training_size", 0)),
        "use_slice": bool(r.get("use_slice", False)),
        "num_vars": int(r.get("num_vars", 0)),
        "test_size": int(len(r["test_x_ab"])) if r.get("test_x_ab") is not None else 0,
    }

    # root cause
    rc = r.get("root_cause_results") or {}
    out["root_cause"] = {
        "ac_at": [float(x) for x in rc.get("ac_at", [])],
        "ac_star_at": [float(x) for x in rc.get("ac_star_at", [])],
        "avg_at_10": float(rc.get("avg_at_10", 0.0)),
        "avg_star_at_500": float(rc.get("avg_star_at_500", 0.0)),
        "time_tolerance": int(rc.get("time_tolerance", 0)),
        "relaxed_ac_at": [float(x) for x in rc.get("relaxed_ac_at", [])],
        "relaxed_ac_star_at": [float(x) for x in rc.get("relaxed_ac_star_at", [])],
        "relaxed_avg_at_10": float(rc.get("relaxed_avg_at_10", 0.0)),
        "relaxed_avg_star_at_500": float(rc.get("relaxed_avg_star_at_500", 0.0)),
        "predicted_root_causes": rc.get("predicted_root_causes", []),
        "num_vars": int(rc.get("num_vars", 0)),
    }

    # causal discovery（可能为 None）
    cd = r.get("causal_results")
    if cd is not None:
        out["causal_discovery"] = {
            "f1_mean": float(cd.get("f1_mean", 0.0)),
            "f1_std": float(cd.get("f1_std", 0.0)),
            "auroc_mean": float(cd.get("auroc_mean", 0.0)),
            "auroc_std": float(cd.get("auroc_std", 0.0)),
            "auprc_mean": float(cd.get("auprc_mean", 0.0)),
            "auprc_std": float(cd.get("auprc_std", 0.0)),
            "hamming_mean": float(cd.get("hamming_mean", 0.0)),
            "hamming_std": float(cd.get("hamming_std", 0.0)),
            "predicted_causal_matrix": np.asarray(cd["predicted_causal_matrix"]).tolist(),
            "true_causal_matrix": np.asarray(cd["true_causal_matrix"]).tolist(),
        }
    else:
        out["causal_discovery"] = None

    return out


# ============================================================
# 流式数据生成 + 实时异常检测
# ============================================================
STREAM_FAULTS_BY_DATASET: Dict[str, Dict[str, Dict[str, Any]]] = {
    "linear": {
        "equipment_spike": {
        "id": "equipment_spike",
        "name": "设备输出异常升高",
        "category": "设备故障",
        "pattern": "step_up",
        "affected_vars": [1, 2],
        "duration": 8,
        "root_cause": "疑似上游输出异常，导致中间压力与下游响应同步升高。",
        "effect": "相关信号出现阶跃式升高，并可能沿依赖链路传播。",
        "confidence": 0.78,
        },
        "power_loss": {
        "id": "power_loss",
        "name": "局部电源中断",
        "category": "电源故障",
        "pattern": "drop_to_zero",
        "affected_vars": [0, 1, 2, 3],
        "duration": 7,
        "root_cause": "疑似局部供电中断，多路信号同步跌至接近 0。",
        "effect": "多路传感器读数快速下降或归零。",
        "confidence": 0.86,
        },
        "sensor_drift": {
        "id": "sensor_drift",
        "name": "传感器漂移",
        "category": "传感器故障",
        "pattern": "gradual_drift",
        "affected_vars": [2],
        "duration": 16,
        "root_cause": "疑似下游响应传感器发生漂移，读数持续偏离正常范围。",
        "effect": "单路信号缓慢偏移，异常分数逐步累积。",
        "confidence": 0.71,
        },
        "signal_loss": {
        "id": "signal_loss",
        "name": "通信中断/信号丢失",
        "category": "通信故障",
        "pattern": "signal_zero",
        "affected_vars": [3],
        "duration": 10,
        "root_cause": "疑似末端反馈通道通信中断或采集链路丢失。",
        "effect": "单路信号短时间变为 0 或接近 0。",
        "confidence": 0.74,
        },
        "actuator_stuck": {
        "id": "actuator_stuck",
        "name": "执行器卡滞",
        "category": "执行器故障",
        "pattern": "stuck_value",
        "affected_vars": [1],
        "duration": 12,
        "root_cause": "疑似中间压力执行环节卡滞，信号保持在故障发生前的读数附近。",
        "effect": "目标信号失去动态响应，短时间维持固定值。",
        "confidence": 0.69,
        },
        "pressure_drop": {
        "id": "pressure_drop",
        "name": "设备输出异常降低",
        "category": "设备故障",
        "pattern": "step_down",
        "affected_vars": [0, 1],
        "duration": 8,
        "root_cause": "疑似上游输出能力下降，上游输出与中间压力出现同步下跌。",
        "effect": "相关信号出现阶跃式降低。",
        "confidence": 0.76,
        },
    },
    "nonlinear": {
        "nonlinear_load_surge": {
            "id": "nonlinear_load_surge",
            "name": "非线性负载突增",
            "category": "工况扰动",
            "pattern": "step_up",
            "affected_vars": [0, 1, 2],
            "duration": 10,
            "root_cause": "疑似负载输入突增，经耦合压力与温度读数链路放大。",
            "effect": "多路信号出现非平稳升高，异常影响可能被耦合关系放大。",
            "confidence": 0.73,
        },
        "nonlinear_sensor_drift": {
            "id": "nonlinear_sensor_drift",
            "name": "非线性传感器漂移",
            "category": "传感器故障",
            "pattern": "gradual_drift",
            "affected_vars": [2],
            "duration": 18,
            "root_cause": "疑似温度读数传感器缓慢漂移，并被非线性系统放大。",
            "effect": "目标信号逐步偏离正常轨迹，检测分数持续累积。",
            "confidence": 0.7,
        },
        "nonlinear_instability": {
            "id": "nonlinear_instability",
            "name": "耦合系统失稳",
            "category": "系统故障",
            "pattern": "oscillation",
            "affected_vars": [1, 3, 4],
            "duration": 14,
            "root_cause": "疑似耦合压力、振荡响应与执行反馈链路失稳，多个变量出现异常振荡。",
            "effect": "受影响变量呈现短时振荡或放大波动。",
            "confidence": 0.68,
        },
        "nonlinear_signal_loss": {
            "id": "nonlinear_signal_loss",
            "name": "采集通道丢失",
            "category": "通信故障",
            "pattern": "signal_zero",
            "affected_vars": [5],
            "duration": 10,
            "root_cause": "疑似采集通道中断，单路信号短时间归零。",
            "effect": "目标信号变为 0 或接近 0，与邻近变量动态不一致。",
            "confidence": 0.72,
        },
    },
    "swat": {
        "swat_pump_trip": {
            "id": "swat_pump_trip",
            "name": "水泵/流量异常",
            "category": "设备故障",
            "pattern": "step_down",
            "affected_vars": [0, 1, 2],
            "duration": 12,
            "root_cause": "疑似泵停、流量通道异常或出水能力下降。",
            "effect": "流量、压力或液位相关通道出现阶跃式下降。",
            "confidence": 0.81,
        },
        "swat_valve_stuck": {
            "id": "swat_valve_stuck",
            "name": "阀门/执行器卡滞",
            "category": "执行器故障",
            "pattern": "stuck_value",
            "affected_vars": [3],
            "duration": 16,
            "root_cause": "疑似阀门执行器卡滞，阀门开度维持在故障前状态。",
            "effect": "目标通道失去动态响应，短时间保持固定读数。",
            "confidence": 0.78,
        },
        "swat_level_sensor_drift": {
            "id": "swat_level_sensor_drift",
            "name": "液位传感器漂移",
            "category": "传感器故障",
            "pattern": "gradual_drift",
            "affected_vars": [4],
            "duration": 18,
            "root_cause": "疑似液位传感器漂移，读数逐步偏离工艺状态。",
            "effect": "液位相关信号持续上移或下移，异常逐步显现。",
            "confidence": 0.76,
        },
        "swat_quality_sensor_fault": {
            "id": "swat_quality_sensor_fault",
            "name": "水质/压力传感器异常",
            "category": "传感器故障",
            "pattern": "gradual_drift",
            "affected_vars": [2, 4],
            "duration": 18,
            "root_cause": "疑似水质分析仪表或差压测点被设置到异常读数。",
            "effect": "水质、压力或相关工艺信号偏离正常读数。",
            "confidence": 0.74,
        },
    },
}


@app.get("/api/stream/fault-scenarios")
def list_stream_fault_scenarios(dataset_name: str = "linear"):
    return {"faults": list(STREAM_FAULTS_BY_DATASET.get(dataset_name, {}).values())}


def _new_fault_instance(dataset_name: str, fault_id: str, prev: np.ndarray, mul: float) -> Dict[str, Any]:
    spec = STREAM_FAULTS_BY_DATASET[dataset_name][fault_id]
    return {
        "spec": spec,
        "elapsed": 0,
        "duration": int(spec["duration"]),
        "amplitude": float(mul * 2.0),
        "hold_values": prev.astype(float).tolist(),
    }


def _apply_stream_fault(pt: np.ndarray, fault: Dict[str, Any]) -> Dict[str, Any]:
    spec = fault["spec"]
    affected = [int(v) for v in spec["affected_vars"] if int(v) < len(pt)]
    elapsed = int(fault["elapsed"])
    duration = max(1, int(fault["duration"]))
    amp = float(fault["amplitude"])
    progress = min(1.0, (elapsed + 1) / duration)
    pattern = spec["pattern"]

    if pattern == "step_up":
        for rank, v in enumerate(affected):
            pt[v] += amp * max(0.35, 1.0 - rank * 0.25)
    elif pattern == "drop_to_zero":
        pt[affected] = 0.0
    elif pattern == "gradual_drift":
        for v in affected:
            pt[v] += amp * 0.08 * (elapsed + 1)
    elif pattern == "signal_zero":
        pt[affected] = 0.0
    elif pattern == "stuck_value":
        for v in affected:
            pt[v] = float(fault["hold_values"][v])
    elif pattern == "step_down":
        for rank, v in enumerate(affected):
            pt[v] -= amp * max(0.35, 1.0 - rank * 0.25)
    elif pattern == "oscillation":
        for rank, v in enumerate(affected):
            pt[v] += amp * 0.45 * np.sin((elapsed + 1) * 1.35 + rank)

    fault["elapsed"] = elapsed + 1
    return {
        "fault_id": spec["id"],
        "fault_name": spec["name"],
        "fault_category": spec["category"],
        "fault_pattern": spec["pattern"],
        "fault_effect": spec["effect"],
        "fault_root_cause": spec["root_cause"],
        "fault_confidence": spec["confidence"],
        "fault_progress": progress,
        "affected_vars": affected,
        "amplitude": amp,
    }


def _apply_stream_rule_detectors(
    pt: np.ndarray,
    scores: List[float],
    detected: List[bool],
) -> Tuple[List[float], List[bool], List[Dict[str, Any]]]:
    """工程规则兜底：识别模型不一定敏感、但现场含义明确的硬故障形态。"""
    n_vars = len(pt)
    if n_vars == 0:
        return scores, detected, []

    rule_hits: List[Dict[str, Any]] = []
    zero_vars = [int(i) for i, value in enumerate(pt) if abs(float(value)) <= 0.02]
    min_zero_vars = max(3, int(np.ceil(n_vars * 0.75)))

    if len(zero_vars) >= min_zero_vars:
        for v in zero_vars:
            detected[v] = True
            scores[v] = max(float(scores[v] if v < len(scores) else 0.0), 6.0)
        rule_hits.append({
            "id": "multi_channel_zero",
            "name": "多路信号归零规则",
            "reason": "多路测点同时接近 0，符合断电或采集链路中断特征。",
            "affected_vars": zero_vars,
        })

    return scores, detected, rule_hits


def _stream_loop(session: Session, loop: asyncio.AbstractEventLoop) -> None:
    """在线程池中循环生成或回放时间序列数据点并实时推理。每 300ms 发出一个 tick。"""
    import time as _time

    dataset_name = session.dataset_name
    mul = float(session.options.get('mul', 3))
    window_size = int(session.options.get('window_size', 1))
    replay_source: Optional[np.ndarray] = None
    replay_t = 0

    if dataset_name != "linear":
        x_n_list = session.data_class.data_dict.get("x_n_list")
        if x_n_list is not None and len(x_n_list) > 0:
            replay_source = np.asarray(x_n_list[0], dtype=float)

    # 从现有 buffer 末尾恢复状态，或从零开始
    with session._lock:
        if len(session.stream_buffer) > 0:
            prev = np.array(session.stream_buffer[-1], dtype=float)
        else:
            if replay_source is not None and len(replay_source) > 0:
                prev = np.array(replay_source[0], dtype=float)
            else:
                num_vars = int(session.options.get("num_vars", 4))
                prev = np.zeros(num_vars)

    while session.stream_is_running:
        with session._lock:
            stop_after_current_fault = (
                session.stream_stop_requested
                and not session.stream_inject_pending
                and session.stream_active_fault is None
            )
        if stop_after_current_fault:
            break

        if dataset_name == "linear":
            a = session.data_class.data_dict['a']
            eps = 0.4 * np.random.randn(4)
            xp, wp, yp, zp = prev
            pt = np.array([
                a[0] * xp + eps[0],
                a[1] * wp + a[2] * xp + eps[1],
                a[3] * yp + a[4] * wp + eps[2],
                a[5] * zp + a[6] * wp + a[7] * yp + eps[3],
            ])
        elif replay_source is not None and len(replay_source) > 0:
            pt = np.array(replay_source[replay_t % len(replay_source)], dtype=float)
            replay_t += 1
        else:
            pt = prev + 0.1 * np.random.randn(len(prev))

        is_anomaly = False
        is_fault_active = False
        fault_start = False
        anomaly_vars: List[int] = []
        anomaly_amp: float = 0.0
        fault_meta: Optional[Dict[str, Any]] = None

        with session._lock:
            pending_fault_id = session.stream_pending_fault_id if session.stream_inject_pending else None
            if pending_fault_id:
                session.stream_inject_pending = False
                session.stream_pending_fault_id = None
                session.stream_active_fault = _new_fault_instance(dataset_name, pending_fault_id, prev, mul)
                fault_start = True

            active_fault = session.stream_active_fault

        if active_fault is not None:
            fault_meta = _apply_stream_fault(pt, active_fault)
            is_fault_active = True
            is_anomaly = fault_start
            anomaly_vars = [int(v) for v in fault_meta["affected_vars"]]
            anomaly_amp = float(fault_meta["amplitude"])
            if int(active_fault["elapsed"]) >= int(active_fault["duration"]):
                with session._lock:
                    session.stream_active_fault = None

        with session._lock:
            session.stream_buffer.append(pt.tolist())
            session.stream_anomaly_flags.append(is_fault_active)
            buf = list(session.stream_buffer)

        n_vars = int(len(pt))
        scores: List[float] = [0.0] * n_vars
        detected: List[bool] = [False] * n_vars
        inference_ms: Optional[float] = None
        model_detected = False
        if (
            len(buf) >= window_size + 1
            and session.run_status == "done"
            and session.results is not None
        ):
            model = session.results.get("aerca_model")
            if model is not None:
                try:
                    window_arr = np.array(buf[-(window_size + 1):], dtype=np.float32)
                    inference_start = _time.perf_counter()
                    result = model.infer_single_window(window_arr)
                    inference_ms = (_time.perf_counter() - inference_start) * 1000
                    scores = result["scores"]
                    detected = result["detected"]
                    model_detected = any(bool(v) for v in detected)
                except Exception:  # noqa: BLE001
                    pass

        scores, detected, rule_hits = _apply_stream_rule_detectors(pt, scores, detected)
        rule_detected = len(rule_hits) > 0
        if model_detected and rule_detected:
            detection_source = "model+rule"
        elif rule_detected:
            detection_source = "rule"
        elif any(bool(v) for v in detected):
            detection_source = "model"
        else:
            detection_source = None

        msg: Dict[str, Any] = {
            "type": "tick",
            "t": len(buf) - 1,
            "values": pt.tolist(),
            "scores": scores,
            "detected": detected,
            "inference_ms": inference_ms,
            "detection_source": detection_source,
            "rule_hits": rule_hits,
            "is_anomaly_step": is_anomaly,
            "is_fault_active": is_fault_active,
            "fault_start": fault_start,
            "anomaly_vars": anomaly_vars if is_anomaly else None,
            "anomaly_amp": anomaly_amp if is_anomaly else None,
            "fault": fault_meta if fault_meta is not None else None,
        }
        session.push_stream(msg, loop)
        prev = pt
        _time.sleep(0.3)

    with session._lock:
        session.stream_is_running = False
        session.stream_stop_requested = False
        session.stream_inject_pending = False
        session.stream_pending_fault_id = None
        session.stream_active_fault = None
    session.push_stream({"type": "stopped"}, loop)


@app.post("/api/sessions/{sid}/stream/start")
async def stream_start(sid: str):
    s = _require_session(sid)
    if s.dataset_name not in STREAM_FAULTS_BY_DATASET:
        raise HTTPException(400, f"Streaming not supported for dataset: {s.dataset_name}")
    if s.run_status != "done":
        raise HTTPException(400, "Model must be fully trained before starting the stream")
    if s.stream_is_running:
        raise HTTPException(409, "Stream already running")
    s.stream_is_running = True
    s.stream_stop_requested = False
    s.stream_inject_pending = False
    s.stream_pending_fault_id = None
    s.stream_active_fault = None
    s.stream_buffer = []
    s.stream_anomaly_flags = []
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_loop, s, loop)
    return {"status": "started"}


@app.post("/api/sessions/{sid}/stream/inject")
def stream_inject(sid: str, req: Optional[StreamInjectRequest] = None):
    s = _require_session(sid)
    if not s.stream_is_running:
        raise HTTPException(400, "Stream is not running")
    if s.stream_stop_requested:
        raise HTTPException(400, "Stream is stopping; wait for it to finish before injecting another fault")
    fault_id = (req.fault_id if req else None) or str(s.options.get("fault_id") or "equipment_spike")
    dataset_faults = STREAM_FAULTS_BY_DATASET.get(s.dataset_name, {})
    if fault_id not in dataset_faults:
        raise HTTPException(400, f"Unknown fault_id: {fault_id}")
    with s._lock:
        s.stream_inject_pending = True
        s.stream_pending_fault_id = fault_id
    return {"status": "pending", "fault": dataset_faults[fault_id]}


@app.post("/api/sessions/{sid}/stream/stop")
def stream_stop(sid: str):
    s = _require_session(sid)
    if not s.stream_is_running:
        return {"status": "stopped"}
    with s._lock:
        s.stream_stop_requested = True
        has_pending_work = bool(s.stream_inject_pending or s.stream_active_fault is not None)
        if not has_pending_work:
            s.stream_is_running = False
    return {"status": "stopping" if has_pending_work else "stopped"}


@app.websocket("/api/sessions/{sid}/stream/ws")
async def stream_ws(websocket: WebSocket, sid: str):
    await websocket.accept()
    s = store.get(sid)
    if s is None:
        await websocket.send_json({"type": "error", "message": f"Session not found: {sid}"})
        await websocket.close()
        return

    queue = s.add_stream_subscriber()
    try:
        await websocket.send_json({"type": "hello", "is_running": s.stream_is_running})
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(msg)
                if msg.get("type") == "stopped":
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("Stream WebSocket error")
    finally:
        s.remove_stream_subscriber(queue)


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    if not store.delete(sid):
        raise HTTPException(404, "Session not found")
    return {"status": "deleted"}


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": store.list_summary()}


# ============================================================
# 静态托管 React 构建产物（生产期）
# ============================================================
FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    logger.info("Mounted frontend static files from %s", FRONTEND_DIST)
else:
    logger.info("Frontend dist not found at %s; running API-only mode", FRONTEND_DIST)
