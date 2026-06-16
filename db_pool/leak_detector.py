"""
泄漏检测模块
============

职责:
- 启动后台守护线程,周期性遍历 borrowed 集合;
- 对每个已借出连接检查 borrow_seconds,超过 leak_threshold 判定为泄漏;
- 对判定为泄漏的连接:
    * 记录日志(含 borrow_stack,便于定位哪里借了没还);
    * 调用用户注册的 leak_listener 回调(可对接监控告警系统);
    * 若 force_reclaim_leaked=True,从 borrowed 表中移除并强制关闭
      (风险操作,默认关闭;开启前请确认业务上确实能容忍连接被外部回收)。
- 同一连接在 leak_cooldown 时间内只告警一次,避免日志风暴。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Set

from .connection import PooledConnection
from .pool_manager import PoolManager


logger = logging.getLogger("db_pool.leak")


@dataclass
class LeakInfo:
    """一条泄漏告警的结构化信息。"""
    conn_id: int
    borrowed_seconds: float
    borrower_thread: Optional[int]
    stack: str


LeakListener = Callable[[LeakInfo], None]


class LeakDetector:
    """
    连接泄漏检测守护线程。

    典型参数:
    - check_interval = 60s: 每分钟扫一次借出表;
    - leak_threshold = 300s: 借出超过 5 分钟视为可疑;
    - force_reclaim_leaked = False: 仅告警,不强制回收。
    """

    def __init__(
        self,
        manager: PoolManager,
        *,
        check_interval: float = 60.0,
        leak_threshold: float = 300.0,
        leak_cooldown: float = 600.0,
        force_reclaim_leaked: bool = False,
        leak_listener: Optional[LeakListener] = None,
    ) -> None:
        if check_interval <= 0:
            raise ValueError("check_interval must be > 0")
        if leak_threshold <= 0:
            raise ValueError("leak_threshold must be > 0")

        self._manager = manager
        self._check_interval = check_interval
        self._leak_threshold = leak_threshold
        self._leak_cooldown = leak_cooldown
        self._force_reclaim = force_reclaim_leaked
        self._listener = leak_listener

        # conn_id -> 上次告警时间(monotonic)
        self._last_alert: dict[int, float] = {}
        self._alerted: Set[int] = set()

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------- 启停

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="db-pool-leak-detector",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Leak detector started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---------------------------------------------------------- 主循环

    def _run(self) -> None:
        self._stop_evt.wait(min(5.0, self._check_interval))
        while not self._stop_evt.is_set():
            try:
                self._run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Leak detector loop error: %s", exc)
            self._stop_evt.wait(self._check_interval)

    def _run_once(self) -> None:
        now = time.monotonic()
        snapshot = self._manager.snapshot_borrowed()

        # 清理已归还连接的告警记录,防止无限增长
        alive_ids = set(snapshot.keys())
        for cid in list(self._last_alert.keys()):
            if cid not in alive_ids:
                self._last_alert.pop(cid, None)

        leaked_this_round = 0
        for conn_id, conn in snapshot.items():
            if self._stop_evt.is_set():
                break
            borrow_secs = conn.borrow_seconds
            if borrow_secs < self._leak_threshold:
                continue

            # 冷却控制
            last = self._last_alert.get(conn_id, 0.0)
            if now - last < self._leak_cooldown:
                continue
            self._last_alert[conn_id] = now

            info = LeakInfo(
                conn_id=conn_id,
                borrowed_seconds=borrow_secs,
                borrower_thread=conn.borrower_thread,
                stack=conn.borrow_stack or "<stack capture disabled>",
            )
            self._report_leak(info, conn)
            leaked_this_round += 1

            if self._force_reclaim:
                self._reclaim(conn)

        if leaked_this_round:
            self._manager.inc_leaked(leaked_this_round)
            logger.warning(
                "Leak detection round: %d connections leaked (>= %.0fs)",
                leaked_this_round, self._leak_threshold,
            )

    # ---------------------------------------------------------- 告警 & 回收

    def _report_leak(self, info: LeakInfo, conn: PooledConnection) -> None:
        logger.warning(
            "LEAK ALERT: conn#%d borrowed for %.1fs by thread=%s\n"
            "Borrow stack:\n%s\nConnection repr: %r",
            info.conn_id, info.borrowed_seconds, info.borrower_thread,
            info.stack, conn,
        )
        if self._listener is not None:
            try:
                self._listener(info)
            except Exception as exc:  # noqa: BLE001
                logger.exception("leak_listener raised: %s", exc)

    def _reclaim(self, conn: PooledConnection) -> None:
        logger.error("Force-reclaiming leaked connection id=%s", conn.conn_id)
        self._manager.destroy_connection(conn)
