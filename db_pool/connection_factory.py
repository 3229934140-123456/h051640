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

通过依赖注入方式与池解耦: 池不关心具体的 DB-API 驱动,只通过工厂交互。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Dict

from .connection import PooledConnection


logger = logging.getLogger("db_pool.factory")


class ConnectionFactory:
    """
    连接工厂。

    典型用法::

        factory = ConnectionFactory(
            create_fn=lambda: psycopg2.connect(**dsn),
            ping_fn=lambda c: c.cursor().execute("SELECT 1"),
            reset_fn=lambda c: c.rollback(),
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
    ) -> None:
        """
        :param create_fn:    无参函数,返回一个新建的底层数据库连接。
        :param ping_fn:      接受底层连接,执行一次轻量探活查询;抛异常视为连接失效。
                             默认为尝试调用 conn.ping() / conn.cursor().execute("SELECT 1")。
        :param reset_fn:     归还时重置连接状态;默认执行回滚并恢复 autocommit。
        :param destroy_fn:   关闭底层连接;默认调用 conn.close()。
        :param ping_timeout: ping 操作的期望超时(秒),仅作为元信息传递给自定义 ping_fn。
        """
        if not callable(create_fn):
            raise TypeError("create_fn must be callable")

        self._create_fn = create_fn
        self._ping_fn = ping_fn
        self._reset_fn = reset_fn
        self._destroy_fn = destroy_fn
        self._ping_timeout = ping_timeout

    # ------------------------------------------------------------------ 创建

    def create(self, pool_ref: Any) -> PooledConnection:
        """
        创建一个新连接并包装为 PooledConnection。

        预创建、动态扩容、以及失效连接替换都会走这里。
        """
        logger.debug("Creating new database connection")
        real_conn = self._create_fn()
        if real_conn is None:
            raise RuntimeError("create_fn returned None")
        return PooledConnection(real_conn, pool_ref)

    # ------------------------------------------------------------------ 销毁

    def destroy(self, pooled: PooledConnection) -> None:
        """销毁底层连接。可被健康检查、池收缩、池关闭等场景调用。"""
        logger.debug("Destroying connection id=%s", pooled.conn_id)
        try:
            pooled._force_close(self._destroy_fn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error while destroying connection %s: %s", pooled.conn_id, exc)

    # ------------------------------------------------------------------ 探活

    def ping(self, pooled: PooledConnection) -> bool:
        """
        检查连接是否仍然可用。

        顺序:
        1. 若用户提供了 ping_fn,调用它;
        2. 否则尝试 DB-API 常见的 ping() 方法;
        3. 再退化到执行 SELECT 1。
        任何一步抛异常都认为连接已死。
        """
        real = pooled.real_connection
        if real is None:
            return False

        try:
            if self._ping_fn is not None:
                self._ping_fn(real)
            else:
                self._default_ping(real)
            pooled.touch_checked()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("Connection id=%s ping failed: %s", pooled.conn_id, exc)
            return False

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
        归还连接前重置其状态,避免连接"脏"了之后影响下一个借用人。

        返回 True 表示重置成功可放回池,False 表示连接应直接销毁。
        """
        real = pooled.real_connection
        if real is None:
            return False

        try:
            if self._reset_fn is not None:
                self._reset_fn(real)
            else:
                self._default_reset(real)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reset connection id=%s failed, will destroy: %s",
                pooled.conn_id, exc,
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
