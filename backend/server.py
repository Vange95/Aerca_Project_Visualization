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
from typing import Any, Dict, List, Optional

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


class RunModelRequest(BaseModel):
    epochs: int = Field(50, ge=1, le=10000)
    lr: float = Field(1e-3, gt=0)
    training_aerca: bool = True


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

    # adtype 处理
    if info["supports_adtype"] and req.adtype:
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
    return _cb


def _run_model_blocking(session: Session, req: RunModelRequest, loop: asyncio.AbstractEventLoop):
    """在线程池中阻塞执行模型训练 + 测试。"""
    try:
        session.run_status = "running"
        session.run_error = None
        session.progress_log.clear()
        cb = _make_progress_callback(session, loop)
        cb({"phase": "starting", "message": "Initializing model..."})

        # 通过 sys.argv hack 传给 main()
        from main import main as run_main

        argv = ["main.py", "--dataset_name", session.dataset_name]
        info = DATASET_REGISTRY[session.dataset_name]
        if info["supports_adtype"] and "adtype" in session.options:
            argv += ["--adtype", str(session.options["adtype"])]
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
        cb({"phase": "done", "message": "All stages complete."})
    except Exception as e:  # noqa: BLE001
        logger.exception("Model run failed")
        session.run_status = "failed"
        session.run_error = f"{type(e).__name__}: {e}"
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
# 流式数据生成 + 实时异常检测（仅 linear 数据集）
# ============================================================
def _stream_loop(session: Session, loop: asyncio.AbstractEventLoop) -> None:
    """在线程池中循环生成 linear 时间序列数据点并实时推理。每 300ms 发出一个 tick。"""
    import time as _time

    a = session.data_class.data_dict['a']
    mul = float(session.options.get('mul', 3))
    window_size = int(session.options.get('window_size', 1))

    # 从现有 buffer 末尾恢复状态，或从零开始
    with session._lock:
        if len(session.stream_buffer) > 0:
            prev = np.array(session.stream_buffer[-1], dtype=float)
        else:
            prev = np.zeros(4)

    while session.stream_is_running:
        eps = 0.4 * np.random.randn(4)
        xp, wp, yp, zp = prev
        pt = np.array([
            a[0] * xp + eps[0],
            a[1] * wp + a[2] * xp + eps[1],
            a[3] * yp + a[4] * wp + eps[2],
            a[5] * zp + a[6] * wp + a[7] * yp + eps[3],
        ])

        is_anomaly = False
        anomaly_vars: List[int] = []
        anomaly_amp: float = 0.0
        if session.stream_inject_pending:
            session.stream_inject_pending = False
            anomaly_amp = mul * 2.0
            n_feat = np.random.randint(1, 4)
            feat = np.random.choice(4, size=n_feat, replace=False)
            pt[feat] += anomaly_amp
            is_anomaly = True
            anomaly_vars = [int(v) for v in feat]

        with session._lock:
            session.stream_buffer.append(pt.tolist())
            session.stream_anomaly_flags.append(is_anomaly)
            buf = list(session.stream_buffer)

        scores: List[float] = [0.0] * 4
        detected: List[bool] = [False] * 4
        if (
            len(buf) >= window_size + 1
            and session.run_status == "done"
            and session.results is not None
        ):
            model = session.results.get("aerca_model")
            if model is not None:
                try:
                    window_arr = np.array(buf[-(window_size + 1):], dtype=np.float32)
                    result = model.infer_single_window(window_arr)
                    scores = result["scores"]
                    detected = result["detected"]
                except Exception:  # noqa: BLE001
                    pass

        msg: Dict[str, Any] = {
            "type": "tick",
            "t": len(buf) - 1,
            "values": pt.tolist(),
            "scores": scores,
            "detected": detected,
            "is_anomaly_step": is_anomaly,
            "anomaly_vars": anomaly_vars if is_anomaly else None,
            "anomaly_amp": anomaly_amp if is_anomaly else None,
        }
        session.push_stream(msg, loop)
        prev = pt
        _time.sleep(0.3)

    session.push_stream({"type": "stopped"}, loop)


@app.post("/api/sessions/{sid}/stream/start")
async def stream_start(sid: str):
    s = _require_session(sid)
    if s.dataset_name != "linear":
        raise HTTPException(400, "Streaming only supported for the linear dataset")
    if s.run_status != "done":
        raise HTTPException(400, "Model must be fully trained before starting the stream")
    if s.stream_is_running:
        raise HTTPException(409, "Stream already running")
    s.stream_is_running = True
    s.stream_inject_pending = False
    s.stream_buffer = []
    s.stream_anomaly_flags = []
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_loop, s, loop)
    return {"status": "started"}


@app.post("/api/sessions/{sid}/stream/inject")
def stream_inject(sid: str):
    s = _require_session(sid)
    if not s.stream_is_running:
        raise HTTPException(400, "Stream is not running")
    s.stream_inject_pending = True
    return {"status": "pending"}


@app.post("/api/sessions/{sid}/stream/stop")
def stream_stop(sid: str):
    s = _require_session(sid)
    s.stream_is_running = False
    return {"status": "stopped"}


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
