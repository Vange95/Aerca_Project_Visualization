"""数据集注册表：把项目中的 6 个数据集统一封装。

为避免某个数据集模块（如 lotka_volterra 依赖 numba）的 import 错误拖垮整个后端，
这里采用**懒加载**：每个 dataset 只在第一次被使用时才执行真正的 import。
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

# 让 backend 模块能 import 项目根的 datasets / args / models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger("aerca.dataset_registry")


# 数据集元信息（不直接 import 任何 dataset 模块）
_DATASET_META: Dict[str, Dict[str, Any]] = {
    "linear": {
        "module": "datasets.linear", "class_name": "Linear",
        "args_module": "args.linear_args",
        "use_slice": True, "supports_adtype": True,
        "adtypes": ["spike", "step", "causal"],
        "display_name": "简化工业过程仿真",
        "scenario": "4 路传感器存在明确线性耦合，适合演示设备间影响传播和变量级根因定位。",
        "signal_profile": "信号通常围绕稳定工况波动，故障可表现为尖峰、阶跃、传播或归零。",
        "typical_faults": ["设备输出异常", "传感器突变", "电源/通信类故障"],
        "fault_closed_loop": True,
    },
    "lotka_volterra": {
        "module": "datasets.lotka_volterra", "class_name": "LotkaVolterra",
        "args_module": "args.lotka_volterra_args",
        "use_slice": True, "supports_adtype": False,
        "adtypes": ["non_causal"],
        "display_name": "耦合子系统动态仿真",
        "scenario": "捕食者-猎物 ODE 模型，可类比相互制约的生产子系统或库存-消耗动态。",
        "signal_profile": "变量呈周期性耦合振荡，更适合验证强耦合系统中的异常传播。",
        "typical_faults": ["耦合关系失衡", "子系统响应滞后", "周期波动异常"],
        "fault_closed_loop": False,
    },
    "lorenz96": {
        "module": "datasets.lorenz96", "class_name": "Lorenz96",
        "args_module": "args.lorenz96_args",
        "use_slice": True, "supports_adtype": False,
        "adtypes": ["non_causal"],
        "display_name": "复杂耦合动态系统",
        "scenario": "混沌系统仿真，适合表达高敏感、多变量强耦合的复杂运行环境。",
        "signal_profile": "信号波动强、传播快，适合测试模型在复杂动态下的鲁棒性。",
        "typical_faults": ["复杂工况扰动", "局部异常扩散", "高敏感系统失稳"],
        "fault_closed_loop": False,
    },
    "msds": {
        "module": "datasets.msds", "class_name": "MSDS",
        "args_module": "args.msds_args",
        "use_slice": False, "supports_adtype": False,
        "adtypes": [],
        "display_name": "真实工业传感器监测",
        "scenario": "真实多维工业传感器数据，适合贴近实际监控场景验证异常检测效果。",
        "signal_profile": "包含真实采集噪声和运行状态变化，异常更接近现场传感器表现。",
        "typical_faults": ["设备状态异常", "传感器异常", "工况切换异常"],
        "fault_closed_loop": False,
    },
    "swat": {
        "module": "datasets.swat", "class_name": "SWaT",
        "args_module": "args.swat_args",
        "use_slice": False, "supports_adtype": False,
        "adtypes": [],
        "display_name": "水处理控制系统",
        "scenario": "真实水处理工业控制系统数据，包含泵、阀门、液位、流量等过程变量。",
        "signal_profile": "信号具备明确工艺含义，适合演示水处理设备故障或控制攻击导致的异常。",
        "typical_faults": ["阀门/泵异常", "液位异常", "控制系统攻击"],
        "fault_closed_loop": True,
    },
    "nonlinear": {
        "module": "datasets.nonlinear", "class_name": "Nonlinear",
        "args_module": "args.nonlinear_args",
        "use_slice": True, "supports_adtype": False,
        "adtypes": ["non_causal"],
        "display_name": "非线性设备耦合仿真",
        "scenario": "合成非线性系统，适合模拟压力、流量、温度等非线性耦合过程。",
        "signal_profile": "变量关系不是简单线性，故障可能表现为非对称、非平稳或放大效应。",
        "typical_faults": ["非线性工况漂移", "阀门卡滞", "负载突变"],
        "fault_closed_loop": True,
    },
}


class _LazyDatasetEntry:
    """惰性持有 class 和 args parser。"""

    def __init__(self, name: str, meta: Dict[str, Any]) -> None:
        self.name = name
        self._meta = meta
        self._cls: Optional[type] = None
        self._args_fn: Optional[Callable] = None
        self._import_error: Optional[str] = None

    @property
    def use_slice(self) -> bool:
        return bool(self._meta["use_slice"])

    @property
    def supports_adtype(self) -> bool:
        return bool(self._meta["supports_adtype"])

    @property
    def adtypes(self) -> List[str]:
        return list(self._meta["adtypes"])

    def _ensure_loaded(self) -> None:
        if self._cls is not None and self._args_fn is not None:
            return
        try:
            mod = importlib.import_module(self._meta["module"])
            self._cls = getattr(mod, self._meta["class_name"])
            args_mod = importlib.import_module(self._meta["args_module"])
            self._args_fn = getattr(args_mod, "create_arg_parser")
            self._import_error = None
        except Exception as e:  # noqa: BLE001
            self._import_error = f"{type(e).__name__}: {e}"
            logger.warning("Lazy import failed for %s: %s", self.name, self._import_error)
            raise

    @property
    def cls(self):
        self._ensure_loaded()
        return self._cls

    @property
    def args_fn(self):
        self._ensure_loaded()
        return self._args_fn

    @property
    def import_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._cls is not None,
            "import_error": self._import_error,
        }

    # 兼容旧字典访问
    def __getitem__(self, key: str):
        if key == "use_slice":
            return self.use_slice
        if key == "supports_adtype":
            return self.supports_adtype
        if key == "adtypes":
            return self.adtypes
        if key == "class":
            return self.cls
        if key == "args":
            return self.args_fn
        raise KeyError(key)

    def __contains__(self, key: str):
        return key in {"use_slice", "supports_adtype", "adtypes", "class", "args"}


DATASET_REGISTRY: Dict[str, _LazyDatasetEntry] = {
    name: _LazyDatasetEntry(name, meta) for name, meta in _DATASET_META.items()
}


def list_datasets():
    return [
        {
            "name": e.name,
            "use_slice": e.use_slice,
            "supports_adtype": e.supports_adtype,
            "adtypes": e.adtypes,
            "loaded": e._cls is not None,  # 仅供调试
            "display_name": e._meta.get("display_name", e.name),
            "scenario": e._meta.get("scenario", ""),
            "signal_profile": e._meta.get("signal_profile", ""),
            "typical_faults": e._meta.get("typical_faults", []),
            "fault_closed_loop": bool(e._meta.get("fault_closed_loop", False)),
        }
        for e in DATASET_REGISTRY.values()
    ]


def get_default_options(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    parser = DATASET_REGISTRY[dataset_name].args_fn()
    args = parser.parse_args([])
    return vars(args)
