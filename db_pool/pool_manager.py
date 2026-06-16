"""
池管理核心模块
==============

职责:
- 维护空闲连接集合 (idle)、活跃借出集合 (borrowed)
- 维护总连接数,保证 min_size <= total <= max_size
- 初始化时预创建 min_size 个空闲连接 (warm-up)
- 提供"获取一个空闲连接 / 注册一个新连接 / 移除连接"等原子操作
- 负载统计与动态伸缩决策 (scale up / scale down)

本模块只负责状态,不做任何阻塞等待逻辑 — 等待队列与超时在 borrow_return 模块。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from .connection import PooledConnection
from .connection_factory import ConnectionFactory
from .time_window_stats import TimeWindowStats


logger = logging.getLogger("db_pool.manager")


@dataclass
class PoolStats:
    """池运行时统计快照。"""
    min_size: int = 0
    max_size: int = 0
    total: int = 0
    idle: int = 0
    borrowed: int = 0
    waiting: int = 0
    created: int = 0
    destroyed: int = 0
    borrowed_total: int = 0
    timeouts: int = 0
    leaked: int = 0

    # --- 失败计数
    create_failures: int = 0
    ping_failures: int = 0
    reset_failures: int = 0
    retried_operations: int = 0

    # --- 并发状态量
    pending_creates: int = 0
    in_health_check: int = 0

    # --- 等待耗时 (滑动窗口)
    avg_wait_seconds: float = 0.0
    p99_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0

    # --- 轮换统计
    rotated_total: int = 0


class PoolManager:
    """
    连接池的核心状态管理器。

    - 内部用一把 reentrant 锁保护 idle / borrowed / stats。
    - 空闲集合使用 deque,按 LIFO 取最近用过的(更可能还活着),FIFO 归还。
    - 提供 pop_idle / add_idle / register_borrowed / unregister_borrowed 等原语,
      由借还逻辑模块组合使用。
    - 健康检查移走的空闲连接做探活时,会占用 _in_health_check 计数,使得
      total_count = idle + borrowed + in_health_check,
      从而守住 max_size 不被突破。
    """

    def __init__(
        self,
        factory: ConnectionFactory,
        *,
        min_size: int = 5,
        max_size: int = 30,
        pool_ref: Optional[object] = None,
    ) -> None:
        if not isinstance(factory, ConnectionFactory):
            raise TypeError("factory must be ConnectionFactory")
        if min_size < 0:
            raise ValueError("min_size must be >= 0")
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        if min_size > max_size:
            raise ValueError("min_size must not exceed max_size")

        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._pool_ref = pool_ref

        self._lock = threading.RLock()
        self._idle: Deque[PooledConnection] = deque()
        self._borrowed: Dict[int, PooledConnection] = {}

        self._stats = PoolStats(min_size=min_size, max_size=max_size)
        self._shutdown = False

        # 健康检查从 idle 移走的连接数 (用于计算 total 时要加上,守住 max_size
        self._in_health_check: int = 0

        # 等待耗时统计 (滑动窗口 60s / 2000 样本
        self._wait_stats = TimeWindowStats(window_seconds=60.0, max_samples=2000)

    # ---------------------------------------------------------- 基础属性

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown

    @property
    def total_count(self) -> int:
        """总连接数 = 空闲 + 已借出 + 健康检查正在探活的。"""
        with self._lock:
            return len(self._idle) + len(self._borrowed) + self._in_health_check

    @property
    def idle_count(self) -> int:
        with self._lock:
            return len(self._idle)

    @property
    def borrowed_count(self) -> int:
        with self._lock:
            return len(self._borrowed)

    @property
    def in_health_check_count(self) -> int:
        with self._lock:
            return self._in_health_check

    @property
    def can_create_more(self) -> bool:
        """创建名额是否还有?考虑 idle+borrowed+in_health_check 都不能超过 max_size。
        """
        with self._lock:
            total = len(self._idle) + len(self._borrowed) + self._in_health_check
            return total < self._max_size

    # ---------------------------------------------------------- 初始化预热

    def warm_up(self) -> None:
        """
        预创建 min_size 个连接放入空闲集合。
        创建失败会打 warning 但不抛异常,避免应用启动被数据库抖动阻断。
        """
        if self._min_size <= 0:
            return
        created = 0
        for _ in range(self._min_size):
            try:
                conn = self._factory.create(self._pool_ref)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pre-create connection failed: %s", exc)
                # 注意: create_failures 由 ConnectionFactory 通过 stats_callback 报告
                continue
            with self._lock:
                self._idle.append(conn)
                self._stats.created += 1
            created += 1
        logger.info("Pool warm-up complete: %d/%d pre-created", created, self._min_size)

    # ---------------------------------------------------------- 空闲连接操作

    def pop_idle(self) -> Optional[PooledConnection]:
        """
        从空闲集合栈顶弹出一个连接(LIFO: 取最近归还/使用的)。
        调用者需自行做健康校验,失效的要 replace_idle。
        """
        with self._lock:
            if not self._idle:
                return None
            return self._idle.pop()

    def add_idle(self, conn: PooledConnection) -> None:
        """把已重置好的连接放回空闲集合。"""
        with self._lock:
            self._idle.append(conn)

    def peek_all_idle(self) -> list[PooledConnection]:
        """返回当前空闲连接的快照(非拷贝,仅用于只读遍历如健康检查)。"""
        with self._lock:
            return list(self._idle)

    def remove_idle(self, conn: PooledConnection) -> bool:
        """从空闲集合里移除指定连接(健康检查剔除假死时用)。"""
        with self._lock:
            try:
                self._idle.remove(conn)
                return True
            except ValueError:
                return False

    def mark_health_check_take(self, conn: PooledConnection) -> bool:
        """
        健康检查从 idle 中拿一个连接去探活,先从 idle 移除并计入 in_health_check。
        返回 True 表示成功从 idle 中取出并占坑完成。
        """
        with self._lock:
            try:
                self._idle.remove(conn)
                self._in_health_check += 1
                return True
            except ValueError:
                return False

    def mark_health_check_return(self, conn: PooledConnection, still_alive: bool) -> None:
        """
        健康检查完成,把连接要么放回 idle,要么销毁。无论哪种都要递减 in_health_check。
        """
        with self._lock:
            self._in_health_check = max(0, self._in_health_check - 1)
        if still_alive:
            conn.touch_checked()
            self.add_idle(conn)
        else:
            self.destroy_connection(conn)

    def replace_idle(self, bad: PooledConnection) -> Optional[PooledConnection]:
        """
        用一个新建的好连接"顶替"掉空闲/刚取出但已坏的连接。
        返回新连接(如果还能创建),同时销毁坏连接。
        """
        self._factory.destroy(bad)
        with self._lock:
            self._stats.destroyed += 1
            total = len(self._idle) + len(self._borrowed) + self._in_health_check
            if total >= self._max_size:
                return None
        try:
            new_conn = self._factory.create(self._pool_ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Replace bad connection failed to create new one: %s", exc)
            # 注意: create_failures 由 ConnectionFactory 通过 stats_callback 报告
            return None
        with self._lock:
            self._stats.created += 1
        return new_conn

    # ---------------------------------------------------------- 借出集合操作

    def register_borrowed(self, conn: PooledConnection) -> None:
        """把连接登记到"已借出"表。"""
        with self._lock:
            self._borrowed[conn.conn_id] = conn
            self._stats.borrowed_total += 1

    def unregister_borrowed(self, conn: PooledConnection) -> Optional[PooledConnection]:
        """从"已借出"表移除,返回该连接或 None。"""
        with self._lock:
            return self._borrowed.pop(conn.conn_id, None)

    def snapshot_borrowed(self) -> Dict[int, PooledConnection]:
        """借出表的只读快照,供泄漏检测遍历。"""
        with self._lock:
            return dict(self._borrowed)

    # ---------------------------------------------------------- 创建 / 销毁

    def try_create_one(self) -> Optional[PooledConnection]:
        """
        若尚未达 max_size,创建一个新连接并计入统计。
        否则返回 None。
        """
        with self._lock:
            total = len(self._idle) + len(self._borrowed) + self._in_health_check
            if total >= self._max_size:
                return None
        try:
            conn = self._factory.create(self._pool_ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create new connection: %s", exc)
            # 注意: create_failures 由 ConnectionFactory 通过 stats_callback 报告
            return None
        with self._lock:
            self._stats.created += 1
        return conn

    def destroy_connection(self, conn: PooledConnection) -> None:
        """销毁连接并从所有集合中移除。"""
        with self._lock:
            self._borrowed.pop(conn.conn_id, None)
            try:
                self._idle.remove(conn)
            except ValueError:
                pass
        self._factory.destroy(conn)
        with self._lock:
            self._stats.destroyed += 1

    # ---------------------------------------------------------- 动态伸缩

    def resize(self, new_min: Optional[int] = None, new_max: Optional[int] = None) -> None:
        """
        运行时动态调整 min_size / max_size。
        缩小 max 时不会主动踢掉在用连接,等它们归还后再销毁多余的。
        若扩大 min,且当前总连接数不足,立即创建补齐。
        """
        with self._lock:
            if new_min is not None:
                if new_min < 0:
                    raise ValueError("new_min must be >= 0")
                self._min_size = new_min
                self._stats.min_size = new_min
            if new_max is not None:
                if new_max <= 0:
                    raise ValueError("new_max must be > 0")
                if new_max < self._min_size:
                    raise ValueError("new_max must be >= min_size")
                self._max_size = new_max
                self._stats.max_size = new_max

        need = self._min_size - self.total_count
        for _ in range(max(0, need)):
            c = self.try_create_one()
            if c is None:
                break
            self.add_idle(c)

    def shrink_if_needed(self, max_idle_seconds: float, shrink_target: Optional[int] = None) -> int:
        """
        伸缩中的缩容步骤: 销毁那些闲置超过 max_idle_seconds 且超出 min_size 的连接。
        返回实际销毁的数量。
        """
        target = shrink_target if shrink_target is not None else self._min_size
        removed = 0
        with self._lock:
            total = len(self._idle) + len(self._borrowed) + self._in_health_check
            if total <= target:
                return 0
            now = time.monotonic()
            candidates = [c for c in self._idle if (now - c.last_used_at) > max_idle_seconds]
            excess = total - target
            to_remove = candidates[:excess]
            for c in to_remove:
                try:
                    self._idle.remove(c)
                except ValueError:
                    continue
                self._factory.destroy(c)
                self._stats.destroyed += 1
                removed += 1
        if removed:
            logger.info("Shrank pool: removed %d long-idle connections", removed)
        return removed

    # ---------------------------------------------------------- 统计 & 关闭

    def record_wait_time(self, wait_seconds: float) -> None:
        """记录一次等待耗时,由借还模块在拿到连接后调用。"""
        self._wait_stats.record(wait_seconds)

    def snapshot_stats(self, waiting: int = 0, pending_creates: int = 0) -> PoolStats:
        with self._lock:
            s = PoolStats(
                min_size=self._stats.min_size,
                max_size=self._stats.max_size,
                total=len(self._idle) + len(self._borrowed) + self._in_health_check,
                idle=len(self._idle),
                borrowed=len(self._borrowed),
                waiting=waiting,
                created=self._stats.created,
                destroyed=self._stats.destroyed,
                borrowed_total=self._stats.borrowed_total,
                timeouts=self._stats.timeouts,
                leaked=self._stats.leaked,
                create_failures=self._stats.create_failures,
                ping_failures=self._stats.ping_failures,
                reset_failures=self._stats.reset_failures,
                retried_operations=self._stats.retried_operations,
                rotated_total=self._stats.rotated_total,
                pending_creates=pending_creates,
                in_health_check=self._in_health_check,
                avg_wait_seconds=self._wait_stats.avg(),
                p99_wait_seconds=self._wait_stats.p99(),
                max_wait_seconds=self._wait_stats.max(),
            )
            return s

    def inc_timeout(self) -> None:
        with self._lock:
            self._stats.timeouts += 1

    def inc_leaked(self, n: int = 1) -> None:
        with self._lock:
            self._stats.leaked += n

    def inc_create_failure(self) -> None:
        with self._lock:
            self._stats.create_failures += 1

    def inc_ping_failure(self) -> None:
        with self._lock:
            self._stats.ping_failures += 1

    def inc_reset_failure(self) -> None:
        with self._lock:
            self._stats.reset_failures += 1

    def inc_retried(self, n: int = 1) -> None:
        with self._lock:
            self._stats.retried_operations += n

    def inc_rotated(self, n: int = 1) -> None:
        """连接被轮换(因年龄或使用次数超限)时调用。"""
        with self._lock:
            self._stats.rotated_total += n

    def acquire_lock(self) -> threading.RLock:
        """暴露锁,供借还模块做复合原子操作。"""
        return self._lock

    def begin_shutdown(self) -> None:
        with self._lock:
            self._shutdown = True

    def force_destroy_all(self) -> int:
        """
        关闭池: 先销毁所有空闲连接,再强制销毁仍然在借的(极端情况)。
        返回销毁数量。
        """
        n = 0
        with self._lock:
            while self._idle:
                c = self._idle.pop()
                self._factory.destroy(c)
                self._stats.destroyed += 1
                n += 1
            for c in list(self._borrowed.values()):
                self._factory.destroy(c)
                self._stats.destroyed += 1
                n += 1
            self._borrowed.clear()
        return n

    def waiting_borrowed_count(self) -> int:
        """当归还时,检查是否还有等待者且已借数达到上限,供唤醒。"""
        with self._lock:
            return len(self._borrowed)
