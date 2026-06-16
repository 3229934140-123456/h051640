"""
连接包装模块
============

对底层数据库连接进行封装,附加池管理所需的元信息:
- 创建时间、最后使用时间、最后探活时间
- 借出状态、借用线程、借用堆栈(泄漏检测用)
- 重置与关闭的钩子方法

关键设计:
    归还时,真实连接会从 PooledConnection 中"剥离"(_conn 被置为 None)。
    - 用户持有的旧 PooledConnection 引用会因为 _conn 为空而彻底不可用;
    - 底层真实连接被池回收,包装为新的 PooledConnection 继续复用。
    这样既防止了旧引用滥用,又不影响连接复用。
"""

from __future__ import annotations

import time
import traceback
import threading
from typing import Optional, Any, Callable


class ConnectionReturnedError(RuntimeError):
    """尝试使用已经归还给池的连接。"""


class PooledConnection:
    """
    池化连接包装器。

    透明代理真实连接的所有方法调用,但在连接归还后:
    - 内部 _conn 被置为 None,所有外部访问都会抛 ConnectionReturnedError
    - 真实连接被池回收,通过新的 PooledConnection 实例复用
    """

    __slots__ = (
        "_conn",
        "_pool",
        "_id",
        "_created_at",
        "_last_used_at",
        "_last_checked_at",
        "_borrowed_at",
        "_is_borrowed",
        "_borrower_thread",
        "_borrow_stack",
        "_is_closed",
        "_returned",
        "_lock",
    )

    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, real_conn: Any, pool: Any) -> None:
        with PooledConnection._id_lock:
            PooledConnection._next_id += 1
            self._id = PooledConnection._next_id

        self._conn = real_conn
        self._pool = pool
        now = time.monotonic()
        self._created_at = now
        self._last_used_at = now
        self._last_checked_at = now
        self._borrowed_at = 0.0
        self._is_borrowed = False
        self._borrower_thread: Optional[int] = None
        self._borrow_stack: str = ""
        self._is_closed = False
        self._returned = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 元数据

    @property
    def conn_id(self) -> int:
        return self._id

    @property
    def created_at(self) -> float:
        return self._created_at

    @property
    def last_used_at(self) -> float:
        return self._last_used_at

    @property
    def last_checked_at(self) -> float:
        return self._last_checked_at

    @property
    def borrowed_at(self) -> float:
        return self._borrowed_at

    @property
    def is_borrowed(self) -> bool:
        return self._is_borrowed

    @property
    def borrower_thread(self) -> Optional[int]:
        return self._borrower_thread

    @property
    def borrow_stack(self) -> str:
        return self._borrow_stack

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self._created_at

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used_at

    @property
    def borrow_seconds(self) -> float:
        if not self._is_borrowed:
            return 0.0
        return time.monotonic() - self._borrowed_at

    @property
    def real_connection(self) -> Any:
        self._check_usable()
        return self._conn

    def _peek_real_connection(self) -> Any:
        """
        内部使用: 不做可用性检查,直接取 _conn。
        供池内部做 ping / reset / destroy 用。
        """
        return self._conn

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def is_returned(self) -> bool:
        return self._returned

    def touch_used(self) -> None:
        self._check_usable()
        self._last_used_at = time.monotonic()

    def touch_checked(self) -> None:
        self._last_checked_at = time.monotonic()

    def _check_usable(self) -> None:
        if self._returned or self._conn is None:
            raise ConnectionReturnedError(
                f"PooledConnection #{self._id} has been returned to the pool "
                "and is no longer usable. Please acquire a new connection."
            )

    # ------------------------------------------------------------------ 借出/归还标记

    def mark_borrowed(self, capture_stack: bool = False) -> None:
        """标记为已借出,记录线程和调用栈(用于泄漏检测)。"""
        with self._lock:
            self._is_borrowed = True
            self._borrowed_at = time.monotonic()
            self._borrower_thread = threading.get_ident()
            self._last_used_at = self._borrowed_at
            self._is_closed = False
            if capture_stack:
                self._borrow_stack = "".join(traceback.format_stack()[:-2])
            else:
                self._borrow_stack = ""

    def mark_returned(self) -> None:
        """标记为已归还。"""
        with self._lock:
            self._is_borrowed = False
            self._borrowed_at = 0.0
            self._borrower_thread = None
            self._borrow_stack = ""

    def strip_real_connection(self) -> Any:
        """
        归还时由池调用: 剥离底层真实连接,当前包装对象标记为已归还,不可再用。
        返回剥离出的真实连接,供池重新包装复用或销毁。
        """
        with self._lock:
            real_conn = self._conn
            self._conn = None
            self._returned = True
            self._is_closed = True
            self._is_borrowed = False
            self._pool = None
            return real_conn

    # ------------------------------------------------------------------ 透明代理

    def __getattr__(self, item: str) -> Any:
        """将所有未定义的属性/方法代理给真实连接。"""
        if item.startswith("_"):
            raise AttributeError(item)
        self._check_usable()
        attr = getattr(self._conn, item)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                self._check_usable()
                self._last_used_at = time.monotonic()
                return attr(*args, **kwargs)
            return _wrapped
        return attr

    # ------------------------------------------------------------------ 归还 & 关闭

    def close(self) -> None:
        """用户调用 close() 时将连接归还池中,而非真正关闭。"""
        if self._returned:
            return
        if self._is_closed:
            return
        self._is_closed = True
        if self._pool is not None:
            self._pool._return_connection(self)

    def _force_close(self, close_fn: Optional[Callable[[Any], None]] = None) -> None:
        """
        真正关闭底层连接(池内部销毁时调用)。
        注意: 调用时 _conn 必须还没被 strip。
        """
        with self._lock:
            if self._conn is None:
                return
            try:
                if close_fn:
                    close_fn(self._conn)
                else:
                    close_method = getattr(self._conn, "close", None)
                    if callable(close_method):
                        close_method()
            finally:
                self._conn = None
                self._is_closed = True
                self._returned = True

    def __enter__(self) -> "PooledConnection":
        self._check_usable()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._returned:
            status = "returned"
        elif self._is_borrowed:
            status = "borrowed"
        else:
            status = "idle"
        return (
            f"<PooledConnection id={self._id} status={status} "
            f"age={self.age_seconds:.1f}s>"
        )
