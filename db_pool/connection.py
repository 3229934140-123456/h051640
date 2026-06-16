"""
连接包装模块
============

对底层数据库连接进行封装,附加池管理所需的元信息:
- 创建时间、最后使用时间、最后探活时间
- 借出状态、借用线程、借用堆栈(泄漏检测用)
- 重置与关闭的钩子方法
"""

from __future__ import annotations

import time
import traceback
import threading
from typing import Optional, Any, Callable


class PooledConnection:
    """
    池化连接包装器。

    封装一个真实的数据库连接对象,并维护连接在池中的生命周期元数据。
    不直接对外暴露真实连接,而是通过 __getattr__ 透明代理方法调用,
    并在归还时通过 close() 触发归还逻辑。
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
        return self._conn

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def touch_used(self) -> None:
        self._last_used_at = time.monotonic()

    def touch_checked(self) -> None:
        self._last_checked_at = time.monotonic()

    # ------------------------------------------------------------------ 借出/归还标记

    def mark_borrowed(self, capture_stack: bool = False) -> None:
        """标记为已借出,记录线程和调用栈(用于泄漏检测)。"""
        with self._lock:
            self._is_borrowed = True
            self._borrowed_at = time.monotonic()
            self._borrower_thread = threading.get_ident()
            self._last_used_at = self._borrowed_at
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

    # ------------------------------------------------------------------ 透明代理

    def __getattr__(self, item: str) -> Any:
        """将所有未定义的属性/方法代理给真实连接。"""
        if item.startswith("_"):
            raise AttributeError(item)
        if self._is_closed:
            raise RuntimeError("PooledConnection has been closed and returned to pool")
        attr = getattr(self._conn, item)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                self.touch_used()
                return attr(*args, **kwargs)
            return _wrapped
        return attr

    # ------------------------------------------------------------------ 归还 & 关闭

    def close(self) -> None:
        """用户调用 close() 时将连接归还池中,而非真正关闭。"""
        if self._is_closed:
            return
        self._is_closed = True
        if self._pool is not None:
            self._pool._return_connection(self)

    def _force_close(self, close_fn: Optional[Callable[[Any], None]] = None) -> None:
        """真正关闭底层连接(池内部销毁时调用)。"""
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

    def __enter__(self) -> "PooledConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "borrowed" if self._is_borrowed else "idle"
        return (
            f"<PooledConnection id={self._id} status={status} "
            f"age={self.age_seconds:.1f}s>"
        )
