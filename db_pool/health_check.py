"""
健康检查模块
============

职责:
- 启动后台守护线程,周期性遍历 idle 集合:
  * 对"距离上次探活超过 check_interval"的连接执行 ping;
  * ping 失败的视为假死(数据库单方面关闭/网络断开),从空闲集合剔除并销毁;
  * 同时把"连接总年龄超过 max_lifetime"的老连接主动淘汰,避免数据库服务端
    因连接时间过长而踢掉,产生难以排查的偶发错误。
- 每次探活周期末尾,调用一次 manager.shrink_if_needed 完成缩容(可选)。
- 与池共用同一把锁的只读遍历,保证线程安全;ping 本身在锁外进行,
  避免 I/O 阻塞其他借还操作。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .connection import PooledConnection
from .connection_factory import ConnectionFactory
from .pool_manager import PoolManager


logger = logging.getLogger("db_pool.health")


class HealthChecker:
    """
    后台健康检查守护线程。

    典型策略:
    - check_interval = 30s: 每 30 秒扫一轮,同一连接至少 30s 才会被 ping 一次;
    - idle_before_check = 10s: 刚归还不到 10 秒的连接认为"热乎的",不探活;
    - max_lifetime = 1800s: 连接存活超过 30 分钟就算"寿终正寝",主动销毁;
    - enable_shrink = True: 每轮末尾尝试缩容至 min_size。
    """

    def __init__(
        self,
        manager: PoolManager,
        factory: ConnectionFactory,
        *,
        check_interval: float = 30.0,
        idle_before_check: float = 10.0,
        max_lifetime: float = 1800.0,
        enable_shrink: bool = True,
        shrink_idle_seconds: float = 300.0,
    ) -> None:
        if check_interval <= 0:
            raise ValueError("check_interval must be > 0")

        self._manager = manager
        self._factory = factory
        self._check_interval = check_interval
        self._idle_before_check = idle_before_check
        self._max_lifetime = max_lifetime
        self._enable_shrink = enable_shrink
        self._shrink_idle_seconds = shrink_idle_seconds

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------- 启停

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="db-pool-health-check",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Health checker started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---------------------------------------------------------- 主循环

    def _run(self) -> None:
        # 启动时先 sleep 一下,避免 warm-up 还没完成就开始检查
        self._stop_evt.wait(min(1.0, self._check_interval))
        while not self._stop_evt.is_set():
            try:
                self._run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Health checker loop error: %s", exc)
            self._stop_evt.wait(self._check_interval)

    def _run_once(self) -> None:
        now = time.monotonic()

        # 1) 拿出当前 idle 集合的只读快照
        idle_snapshot = self._manager.peek_all_idle()
        to_check: list[PooledConnection] = []
        to_retire_age: list[PooledConnection] = []

        for c in idle_snapshot:
            age = now - c.created_at
            idle_time = now - c.last_used_at
            since_checked = now - c.last_checked_at

            if self._max_lifetime > 0 and age > self._max_lifetime:
                to_retire_age.append(c)
                continue
            if idle_time < self._idle_before_check:
                continue
            if since_checked < self._check_interval:
                continue
            to_check.append(c)

        # 2) 淘汰超龄连接(从 idle 移除, 然后销毁)
        retired = 0
        for c in to_retire_age:
            if self._manager.remove_idle(c):
                self._manager.destroy_connection(c)
                retired += 1
        if retired:
            logger.info("Retired %d connections due to max_lifetime", retired)

        # 3) 对候选连接逐个 ping(在锁外执行,避免 I/O 阻塞)
        dead = 0
        for c in to_check:
            if self._stop_evt.is_set():
                break
            # 先 double-check: 连接在我们 ping 之前可能已经被借走了
            # 做法: 尝试从 idle 中 remove,成功才是我们的
            if not self._manager.remove_idle(c):
                continue
            if self._factory.ping(c):
                c.touch_checked()
                self._manager.add_idle(c)
            else:
                logger.info(
                    "Connection id=%s is dead (idle for %.0fs, age %.0fs), removing",
                    c.conn_id, c.idle_seconds, c.age_seconds,
                )
                self._manager.destroy_connection(c)
                dead += 1

        # 4) 缩容
        if self._enable_shrink and not self._manager.is_shutdown:
            self._manager.shrink_if_needed(max_idle_seconds=self._shrink_idle_seconds)

        # 5) 淘汰/缩容后若低于 min_size, 补回新连接(保证 min_size 基线)
        if not self._manager.is_shutdown:
            self._top_up_min_size()

        if dead or retired:
            logger.debug(
                "Health check round: dead=%d retired=%d remaining_idle=%d",
                dead, retired, self._manager.idle_count,
            )

    def _top_up_min_size(self) -> None:
        """补齐 min_size。创建失败不计入健康检查失败,仅打 warning。"""
        while not self._manager.is_shutdown and self._manager.can_create_more:
            if self._manager.total_count >= self._manager.min_size:
                break
            new_c = self._manager.try_create_one()
            if new_c is None:
                break
            self._manager.add_idle(new_c)
