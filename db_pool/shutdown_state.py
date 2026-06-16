"""
关闭状态模块
============

定义连接池关闭过程中的各阶段状态,方便外部监控停机流程:

    NOT_SHUTTING -> BEGIN_GRACEFUL_WAIT -> GRACEFUL_WAIT_TIMEOUT
        -> FORCE_DESTROY -> SHUTDOWN_COMPLETE

同时记录各阶段的关键指标: 开始时间、等待时长、强制销毁数量、
仍未归还的连接列表(及对应的调用栈)。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ShutdownPhase(str, Enum):
    NOT_SHUTTING = "not_shutting"
    STOPPING_BACKGROUND = "stopping_background"
    BEGIN_GRACEFUL_WAIT = "begin_graceful_wait"
    GRACEFUL_WAIT_TIMEOUT = "graceful_wait_timeout"
    FORCE_DESTROY = "force_destroy"
    SHUTDOWN_COMPLETE = "shutdown_complete"


@dataclass
class ShutdownInfo:
    """关闭过程的只读快照。"""
    phase: ShutdownPhase = ShutdownPhase.NOT_SHUTTING
    started_at: Optional[float] = None
    graceful_wait_deadline: Optional[float] = None
    graceful_wait_elapsed: float = 0.0
    waiting_for_borrowed: int = 0
    force_destroyed_count: int = 0
    error: Optional[str] = None
    completed_at: Optional[float] = None
    # 仍未归还的 conn_id 列表(仅在 wait_timeout 时填充)
    outstanding_conn_ids: List[int] = field(default_factory=list)

    @property
    def is_shutting(self) -> bool:
        return self.phase not in (
            ShutdownPhase.NOT_SHUTTING, ShutdownPhase.SHUTDOWN_COMPLETE
        )

    @property
    def is_complete(self) -> bool:
        return self.phase == ShutdownPhase.SHUTDOWN_COMPLETE


class ShutdownState:
    """线程安全的关闭状态机。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = ShutdownPhase.NOT_SHUTTING
        self._started_at: Optional[float] = None
        self._graceful_wait_deadline: Optional[float] = None
        self._force_destroyed_count = 0
        self._error: Optional[str] = None
        self._completed_at: Optional[float] = None
        self._outstanding_conn_ids: List[int] = []

    def set_phase(self, phase: ShutdownPhase) -> None:
        with self._lock:
            if phase == ShutdownPhase.STOPPING_BACKGROUND and \
                    self._phase == ShutdownPhase.NOT_SHUTTING:
                self._started_at = time.monotonic()
            self._phase = phase
            if phase == ShutdownPhase.SHUTDOWN_COMPLETE:
                self._completed_at = time.monotonic()

    def set_graceful_deadline(self, timeout: float) -> None:
        with self._lock:
            self._graceful_wait_deadline = time.monotonic() + timeout

    def inc_force_destroyed(self, n: int = 1) -> None:
        with self._lock:
            self._force_destroyed_count += n

    def set_error(self, msg: str) -> None:
        with self._lock:
            self._error = msg

    def set_outstanding_conn_ids(self, conn_ids: List[int]) -> None:
        """保存未归还的连接 ID 列表。"""
        with self._lock:
            self._outstanding_conn_ids = list(conn_ids)

    def snapshot(
        self,
        borrowed_count: int = 0,
        outstanding_conn_ids: Optional[List[int]] = None,
    ) -> ShutdownInfo:
        with self._lock:
            now = time.monotonic()
            elapsed = 0.0
            if self._started_at is not None:
                if self._completed_at is not None:
                    elapsed = self._completed_at - self._started_at
                else:
                    elapsed = now - self._started_at
            # 如果传入了 outstanding_conn_ids,保存到内部状态
            if outstanding_conn_ids is not None:
                self._outstanding_conn_ids = list(outstanding_conn_ids)
            return ShutdownInfo(
                phase=self._phase,
                started_at=self._started_at,
                graceful_wait_deadline=self._graceful_wait_deadline,
                graceful_wait_elapsed=elapsed,
                waiting_for_borrowed=borrowed_count,
                force_destroyed_count=self._force_destroyed_count,
                error=self._error,
                completed_at=self._completed_at,
                outstanding_conn_ids=list(self._outstanding_conn_ids),
            )
