"""
连接池主入口
============

本模块把 connection_factory / pool_manager / borrow_return / health_check /
leak_detector 组合为对外的单一入口 ConnectionPool,并提供 dataclass 形式的
PoolConfig 便于集中配置。

使用示例::

    import psycopg2
    from db_pool import ConnectionPool, PoolConfig

    cfg = PoolConfig(
        min_size=5, max_size=20,
        borrow_timeout=5, leak_threshold=300,
    )
    pool = ConnectionPool(
        create_fn=lambda: psycopg2.connect(dbname="app"),
        config=cfg,
    )
    pool.start()

    with pool.acquire() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")

    pool.close(graceful=True, wait_timeout=30)
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING

from .connection_factory import ConnectionFactory
from .pool_manager import PoolManager, PoolStats
from .borrow_return import BorrowReturn, GetTimeoutError, PoolClosedError
from .health_check import HealthChecker
from .leak_detector import LeakDetector, LeakInfo, LeakListener


if TYPE_CHECKING:
    from .connection import PooledConnection


logger = logging.getLogger("db_pool")


# --------------------------------------------------------------------------- 配置

@dataclass
class PoolConfig:
    """
    连接池的所有可调参数集中在此。

    尺寸 & 超时:
        min_size:             预热后维持的最小空闲连接数
        max_size:             连接总数上限 (idle+borrowed 均计入)
        borrow_timeout:       acquire 时的等待超时,秒

    借/还校验:
        test_on_borrow:       借出时是否 ping 一次校验可用性
        capture_stack:        借出时是否抓取调用栈(泄漏检测会打印;开销较大)

    健康检查 (后台线程):
        health_check_enabled: 是否启动健康检查线程
        check_interval:       健康检查扫描周期,秒
        idle_before_check:    空闲多久后才参与探活,秒
        max_lifetime:         连接的最大存活时长,秒 (0=不限)
        enable_shrink:        健康检查后是否触发缩容
        shrink_idle_seconds:  闲置多久后可被缩容,秒

    泄漏检测 (后台线程):
        leak_check_enabled:   是否启动泄漏检测线程
        leak_threshold:       借出多久后判定为泄漏,秒
        leak_cooldown:        同一连接两次告警的最小间隔,秒
        force_reclaim_leaked: 发现泄漏是否强制回收连接(高风险!)
        leak_listener:        自定义泄漏告警回调,接收 LeakInfo

    探活/重置钩子(可选):
        ping_fn:              自定义探活函数,参数为底层连接
        reset_fn:             自定义归还重置函数
        destroy_fn:           自定义关闭函数
    """
    min_size: int = 5
    max_size: int = 30
    borrow_timeout: float = 8.0

    test_on_borrow: bool = True
    capture_stack: bool = False

    health_check_enabled: bool = True
    check_interval: float = 30.0
    idle_before_check: float = 10.0
    max_lifetime: float = 1800.0
    enable_shrink: bool = True
    shrink_idle_seconds: float = 300.0

    leak_check_enabled: bool = True
    leak_threshold: float = 300.0
    leak_cooldown: float = 600.0
    force_reclaim_leaked: bool = False
    leak_listener: Optional[LeakListener] = None

    ping_fn: Optional[Callable[[Any], None]] = None
    reset_fn: Optional[Callable[[Any], None]] = None
    destroy_fn: Optional[Callable[[Any], None]] = None


# --------------------------------------------------------------------------- 主类

class ConnectionPool:
    """
    生产级数据库连接池。

    使用模式:
        1. 构造: 传入 create_fn 或 factory + config
        2. start(): 预热连接 + 启动后台线程
        3. acquire() / borrow() 获取连接, with 块结束即归还
        4. close(): 优雅关闭池,等待在用连接归还
    """

    def __init__(
        self,
        *,
        create_fn: Optional[Callable[[], Any]] = None,
        factory: Optional[ConnectionFactory] = None,
        config: Optional[PoolConfig] = None,
    ) -> None:
        if config is None:
            config = PoolConfig()
        self._config = config

        if factory is None:
            if create_fn is None:
                raise ValueError("Either create_fn or factory must be provided")
            factory = ConnectionFactory(
                create_fn=create_fn,
                ping_fn=config.ping_fn,
                reset_fn=config.reset_fn,
                destroy_fn=config.destroy_fn,
            )
        elif create_fn is not None:
            raise ValueError("create_fn and factory are mutually exclusive")
        self._factory = factory

        self._manager = PoolManager(
            self._factory,
            min_size=config.min_size,
            max_size=config.max_size,
            pool_ref=self,
        )

        self._borrow = BorrowReturn(
            self._manager,
            self._factory,
            borrow_timeout=config.borrow_timeout,
            test_on_borrow=config.test_on_borrow,
            capture_stack_on_borrow=config.capture_stack,
        )

        self._health: Optional[HealthChecker] = None
        if config.health_check_enabled:
            self._health = HealthChecker(
                self._manager,
                self._factory,
                check_interval=config.check_interval,
                idle_before_check=config.idle_before_check,
                max_lifetime=config.max_lifetime,
                enable_shrink=config.enable_shrink,
                shrink_idle_seconds=config.shrink_idle_seconds,
            )

        self._leak: Optional[LeakDetector] = None
        if config.leak_check_enabled:
            self._leak = LeakDetector(
                self._manager,
                check_interval=max(0.5, config.leak_threshold / 10),
                leak_threshold=config.leak_threshold,
                leak_cooldown=config.leak_cooldown,
                force_reclaim_leaked=config.force_reclaim_leaked,
                leak_listener=config.leak_listener,
            )

        self._started = False
        self._closed = False
        self._close_lock = threading.Lock()

    # ---------------------------------------------------------- 生命周期

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_closed(self) -> bool:
        return self._closed

    def start(self) -> "ConnectionPool":
        """
        预热连接并启动后台线程。可重复调用,幂等。
        返回 self,方便链式调用: pool = ConnectionPool(...).start()
        """
        if self._started:
            return self
        if self._closed:
            raise PoolClosedError("Cannot start a closed pool")
        self._manager.warm_up()
        if self._health is not None:
            self._health.start()
        if self._leak is not None:
            self._leak.start()
        self._started = True
        logger.info(
            "Connection pool started: min=%d max=%d health=%s leak=%s",
            self._config.min_size, self._config.max_size,
            self._health is not None, self._leak is not None,
        )
        return self

    # ---------------------------------------------------------- 借出接口

    def acquire(self, timeout: Optional[float] = None) -> "PooledConnection":
        """
        获取一个连接(若未启动则自动 start)。使用 close() 或 with 块归还。

        :raises GetTimeoutError: 等待超时
        :raises PoolClosedError: 池已关闭
        """
        if not self._started and not self._closed:
            self.start()
        return self._borrow.borrow(timeout=timeout)

    def borrow(self, timeout: Optional[float] = None) -> "PooledConnection":
        """acquire 的别名。"""
        return self.acquire(timeout=timeout)

    @contextmanager
    def connection(self, timeout: Optional[float] = None) -> Iterator["PooledConnection"]:
        """
        上下文管理器形式获取连接,推荐用法。

        例子::
            with pool.connection() as conn:
                conn.cursor().execute("SELECT 1")
        """
        conn = self.acquire(timeout=timeout)
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ---------------------------------------------------------- 内部归还入口

    def _return_connection(self, conn: "PooledConnection") -> None:
        """PooledConnection.close() 会回调到这里。

        注意: 即便是关闭过程中也走 release(),因为 release() 内部会检查
        manager.is_shutdown 并做正确的销毁 + notify 唤醒优雅关闭的等待者。
        不要在这里做短路判断,否则会丢失 wakeup。
        """
        self._borrow.release(conn)

    # ---------------------------------------------------------- 运行时控制

    def resize(self, new_min: Optional[int] = None, new_max: Optional[int] = None) -> None:
        """运行时调整池大小,立即生效。"""
        self._manager.resize(new_min=new_min, new_max=new_max)

    def stats(self) -> PoolStats:
        """获取当前运行统计快照。"""
        return self._manager.snapshot_stats(waiting=self._borrow.waiting_count)

    # ---------------------------------------------------------- 关闭

    def close(
        self,
        *,
        graceful: bool = True,
        wait_timeout: float = 30.0,
        force: bool = False,
    ) -> None:
        """
        关闭连接池。

        :param graceful:    是否先等待所有在用连接归还 (推荐 True)
        :param wait_timeout:graceful=True 时的等待秒数,超时后仍强制关闭
        :param force:       True = 不等归还,直接全部销毁 (最粗暴)
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        logger.info("Closing connection pool (graceful=%s, wait=%s, force=%s)",
                    graceful, wait_timeout, force)

        # 1) 停止后台线程
        if self._health is not None:
            self._health.stop()
        if self._leak is not None:
            self._leak.stop()

        # 2) 标记池关闭,新的 borrow 直接抛错
        self._manager.begin_shutdown()

        # 3) 优雅等待在用连接归还
        if graceful and not force:
            ok = self._borrow.wait_until_all_returned(timeout=wait_timeout)
            if not ok:
                logger.warning(
                    "Graceful wait timed out after %.1fs, %d connections still borrowed, "
                    "will force-close them",
                    wait_timeout, self._manager.borrowed_count,
                )

        # 4) 销毁剩余所有连接
        destroyed = self._manager.force_destroy_all()
        logger.info("Connection pool closed, %d connections destroyed", destroyed)

    # ---------------------------------------------------------- 上下文

    def __enter__(self) -> "ConnectionPool":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"<ConnectionPool total={s.total} idle={s.idle} borrowed={s.borrowed} "
            f"waiting={s.waiting} closed={self._closed}>"
        )


__all__ = [
    "ConnectionPool",
    "PoolConfig",
    "GetTimeoutError",
    "PoolClosedError",
    "PoolStats",
    "LeakInfo",
]
