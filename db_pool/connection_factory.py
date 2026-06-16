"""
连接工厂模块
============

职责:
- 封装底层数据库连接的创建逻辑 (create_connection)
- 封装连接销毁逻辑 (destroy_connection)
- 在归还连接时重置连接状态 (reset_connection),例如:
  * 回滚未提交的事务
  * 清理会话级临时表/锁/自定义变量
  * 恢复 autocommit 等默认属性
- 提供连接健康性探针 (ping_connection)

新增: 所有操作(create/ping/reset)均支持 RetryPolicy 重试策略,
     失败次数会记录到 PoolStats 中。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .connection import PooledConnection
from .retry import RetryPolicy, run_with_retry, NO_RETRY


logger = logging.getLogger("db_pool.factory")


class ConnectionFactory:
    """
    连接工厂。

    典型用法::

        factory = ConnectionFactory(
            create_fn=lambda: psycopg2.connect(**dsn),
            ping_fn=lambda c: c.cursor().execute("SELECT 1"),
            reset_fn=lambda c: c.rollback(),
            retry_policy=RetryPolicy(max_attempts=2, initial_delay=0.05),
        )
    """

    def __init__(
        self,
        create_fn: Callable[[], Any],
        *,
        ping_fn: Optional[Callable[[Any], None]] = None,
        reset_fn: Optional[Callable[[Any], None]] = None,
        destroy_fn: Optional[Callable[[Any], None]] = None,
        ping_timeout: float = 2.0,
        retry_policy: Optional[RetryPolicy] = None,
        stats_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> None:
        """
        :param create_fn:    无参函数,返回一个新建的底层数据库连接。
        :param ping_fn:      接受底层连接,执行一次轻量探活查询;抛异常视为连接失效。
                             默认为尝试调用 conn.ping() / conn.cursor().execute("SELECT 1")。
        :param reset_fn:     归还时重置连接状态;默认执行回滚并恢复 autocommit。
        :param destroy_fn:   关闭底层连接;默认调用 conn.close()。
        :param ping_timeout: ping 操作的期望超时(秒),仅作为元信息传递给自定义 ping_fn。
        :param retry_policy: 操作失败时的重试策略,默认不重试。
        :param stats_callback:
            失败/重试时的统计回调,签名 (op_name, attempts, total_delay)。
            由 PoolManager 注入,用来更新 PoolStats。
        """
        if not callable(create_fn):
            raise TypeError("create_fn must be callable")

        self._create_fn = create_fn
        self._ping_fn = ping_fn
        self._reset_fn = reset_fn
        self._destroy_fn = destroy_fn
        self._ping_timeout = ping_timeout
        self._retry = retry_policy if retry_policy is not None else NO_RETRY
        self._stats_cb = stats_callback

    # ---------- 内部: 报告统计 ----------
    def _report(self, op: str, attempts: int, delay: float) -> None:
        if self._stats_cb is not None:
            try:
                self._stats_cb(op, attempts, delay)
            except Exception:  # noqa: BLE001
                pass

    # ---------- 内部: 带重试执行 ----------
    def _run(self, op: str, func: Callable[[], Any]):
        failure_count = [0]

        # 每次重试前记录一次失败
        def _on_retry(attempt: int, delay: float, exc: Exception):
            failure_count[0] += 1
            self._report(f"{op}_fail", 1, 0.0)
            # 如果用户自定义了 retry_policy 的 on_retry,也调用它
            if self._retry.on_retry is not None:
                try:
                    self._retry.on_retry(attempt, delay, exc)
                except Exception:  # noqa: BLE001
                    pass

        # 构造一个临时的 policy,把 on_retry 包一下
        import copy
        policy = copy.copy(self._retry)
        policy.on_retry = _on_retry

        outcome = run_with_retry(func, policy)
        if outcome.attempts > 0:
            self._report("retried", outcome.attempts, outcome.total_delay)
        if not outcome.ok:
            # 最后一次失败也要记录
            if failure_count[0] <= outcome.attempts:
                self._report(f"{op}_fail", 1, 0.0)
        return outcome

    # ------------------------------------------------------------------ 创建

    def create(self, pool_ref: Any) -> PooledConnection:
        """
        创建一个新连接并包装为 PooledConnection。
        带重试,失败会记录 create_failures。
        """
        logger.debug("Creating new database connection")

        def _do_create():
            real_conn = self._create_fn()
            if real_conn is None:
                raise RuntimeError("create_fn returned None")
            return PooledConnection(real_conn, pool_ref)

        outcome = self._run("create", _do_create)
        if not outcome.ok:
            raise outcome.last_exc  # type: ignore[misc]
        return outcome.value  # type: ignore[return-value]

    # ------------------------------------------------------------------ 销毁

    def destroy(self, pooled: PooledConnection) -> None:
        """销毁底层连接。可被健康检查、池收缩、池关闭等场景调用。"""
        logger.debug("Destroying connection id=%s", pooled.conn_id)
        try:
            pooled._force_close(self._destroy_fn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error while destroying connection %s: %s", pooled.conn_id, exc)

    def destroy_real(self, real_conn: Any) -> None:
        """直接销毁一个底层连接(剥离后用)。"""
        try:
            if self._destroy_fn:
                self._destroy_fn(real_conn)
            else:
                close_method = getattr(real_conn, "close", None)
                if callable(close_method):
                    close_method()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error while destroying real connection: %s", exc)

    # ------------------------------------------------------------------ 探活

    def ping(self, pooled: PooledConnection) -> bool:
        """
        检查连接是否仍然可用。
        带重试(但通常 ping 失败不重试,因为连接已死)。
        """
        real = pooled._peek_real_connection()
        if real is None:
            return False

        def _do_ping():
            if self._ping_fn is not None:
                self._ping_fn(real)
            else:
                self._default_ping(real)

        outcome = self._run("ping", _do_ping)
        if outcome.ok:
            pooled.touch_checked()
            return True
        logger.info("Connection id=%s ping failed: %s", pooled.conn_id, outcome.last_exc)
        return False

    def ping_real(self, real_conn: Any) -> bool:
        """直接探活一个底层连接(剥离后用)。"""
        if real_conn is None:
            return False

        def _do_ping():
            if self._ping_fn is not None:
                self._ping_fn(real_conn)
            else:
                self._default_ping(real_conn)

        outcome = self._run("ping", _do_ping)
        return outcome.ok

    @staticmethod
    def _default_ping(conn: Any) -> None:
        ping = getattr(conn, "ping", None)
        if callable(ping):
            try:
                ping()
                return
            except TypeError:
                pass

        cur = conn.cursor()
        try:
            cur.execute("SELECT 1")
            cur.fetchall()
        finally:
            try:
                cur.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ 重置

    def reset(self, pooled: PooledConnection) -> bool:
        """
        归还连接前重置其状态。
        带重试,失败会记录 reset_failures。
        """
        real = pooled._peek_real_connection()
        if real is None:
            return False
        return self._reset_real(real, pooled.conn_id)

    def reset_real(self, real_conn: Any, conn_id: int = -1) -> bool:
        """直接重置一个底层连接(剥离后用)。"""
        return self._reset_real(real_conn, conn_id)

    def _reset_real(self, real_conn: Any, conn_id: int) -> bool:
        def _do_reset():
            if self._reset_fn is not None:
                self._reset_fn(real_conn)
            else:
                self._default_reset(real_conn)

        outcome = self._run("reset", _do_reset)
        if outcome.ok:
            return True
        logger.warning(
            "Reset connection id=%s failed, will destroy: %s",
            conn_id, outcome.last_exc,
        )
        return False

    @staticmethod
    def _default_reset(conn: Any) -> None:
        """默认重置: 回滚事务,并尝试恢复 autocommit 为 False。"""
        rollback = getattr(conn, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass

        try:
            conn.autocommit = False
        except (AttributeError, TypeError):
            pass

    # ------------------------------------------------------------------ 包装新连接

    def wrap_real(self, real_conn: Any, pool_ref: Any, borrow_count: int = 0, real_born_at: float = 0.0) -> PooledConnection:
        """
        把一个剥离出来的真实连接重新包装为新的 PooledConnection。
        :param borrow_count: 从旧包装继承的借还次数,用于轮换判断
        :param real_born_at: 从旧包装继承的真实连接出生时间,用于年龄轮换判断
        """
        conn = PooledConnection(real_conn, pool_ref)
        if borrow_count > 0:
            conn.set_borrow_count(borrow_count)
        if real_born_at > 0:
            conn.set_real_born_at(real_born_at)
        return conn
