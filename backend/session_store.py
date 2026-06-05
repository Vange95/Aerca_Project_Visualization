"""In-memory session storage for AERCA backend.

每个会话保存：数据集类实例、生成的数据、模型运行结果及运行状态。
进程内存储；进程退出即丢失（科研演示用，不持久化）。
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    session_id: str
    dataset_name: str
    options: Dict[str, Any]
    data_class: Any  # 数据集对象（持有 data_dict）
    created_at: float = field(default_factory=time.time)

    # 模型运行状态
    run_status: str = "idle"  # idle | running | done | failed | stopped
    run_error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None  # main.main 的返回值（已剥离不可序列化字段）
    training_stop_requested: bool = False

    # 进度日志（每条都会通过 WS 推送）
    progress_log: List[Dict[str, Any]] = field(default_factory=list)

    # WebSocket 订阅者队列
    _subscribers: List[asyncio.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # 流式生成状态（支持线性生成与测试样本回放两种来源）
    stream_is_running: bool = False
    stream_stop_requested: bool = False
    stream_inject_pending: bool = False
    stream_pending_fault_id: Optional[str] = None
    stream_active_fault: Optional[Dict[str, Any]] = None
    stream_buffer: List[Any] = field(default_factory=list)
    stream_anomaly_flags: List[bool] = field(default_factory=list)
    _stream_subscribers: List[asyncio.Queue] = field(default_factory=list)

    def push_stream(self, msg: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """从工作线程调用：把流式 tick 推送到所有流订阅者。"""
        with self._lock:
            subs = list(self._stream_subscribers)
        for q in subs:
            try:
                asyncio.run_coroutine_threadsafe(q.put(msg), loop)
            except Exception:  # noqa: BLE001
                pass

    def add_stream_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2048)
        with self._lock:
            self._stream_subscribers.append(q)
        return q

    def remove_stream_subscriber(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._stream_subscribers.remove(q)
            except ValueError:
                pass

    def push_progress(self, info: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """从工作线程调用：把进度推送到所有订阅者。"""
        with self._lock:
            self.progress_log.append(info)
            subs = list(self._subscribers)
        for q in subs:
            try:
                # 跨线程把消息放入 asyncio.Queue
                asyncio.run_coroutine_threadsafe(q.put(info), loop)
            except Exception:  # noqa: BLE001
                pass

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        with self._lock:
            self._subscribers.append(q)
            # 把已有日志补发，让晚连接的客户端也能看到全部历史
            history = list(self.progress_log)
        for item in history:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                break
        return q

    def remove_subscriber(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, dataset_name: str, options: Dict[str, Any], data_class: Any) -> Session:
        sid = uuid.uuid4().hex[:12]
        session = Session(
            session_id=sid,
            dataset_name=dataset_name,
            options=options,
            data_class=data_class,
        )
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def list_summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "dataset_name": s.dataset_name,
                    "created_at": s.created_at,
                    "run_status": s.run_status,
                }
                for s in self._sessions.values()
            ]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


# 全局单例
store = SessionStore()
