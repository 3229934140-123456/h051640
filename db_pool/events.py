"""
事件订阅系统
============

为连接池提供完整的事件监听能力:
- 连接创建、借出、归还、销毁
- 等待超时、健康检查失败、泄漏告警
- 连接轮换、缩容丢弃

监听器异常完全隔离,单个监听器出错不影响连接池和其他监听器。

使用示例::

    from db_pool import ConnectionPool, PoolConfig, PoolEventType, PoolEvent

    def audit_listener(event: PoolEvent):
        print(f"[{event.type}] conn#{event.conn_id}: {event.details}")

    def alert_listener(event: PoolEvent):
        if event.type in (PoolEventType.HEALTH_CHECK_FAILED,
                          PoolEventType.LEAK_DETECTED,
                          PoolEventType.WAIT_TIMEOUT):
            send_alert(event)

    pool = ConnectionPool(...)
    pool.add_event_listener(audit_listener)
    pool.add_event_listener(alert_listener)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


logger = logging.getLogger("db_pool.events")


class PoolEventType(str, Enum):
    """连接池事件类型枚举。"""

    CONNECTION_CREATED = "connection_created"
    CONNECTION_BORROWED = "connection_borrowed"
    CONNECTION_RETURNED = "connection_returned"
    CONNECTION_DESTROYED = "connection_destroyed"
    CONNECTION_ROTATED = "connection_rotated"

    WAIT_TIMEOUT = "wait_timeout"
    HEALTH_CHECK_FAILED = "health_check_failed"
    LEAK_DETECTED = "leak_detected"

    SHRINK_DISCARDED = "shrink_discarded"
    RESET_FAILED = "reset_failed"
    SHUTDOWN_RETURN = "shutdown_return"


@dataclass
class PoolEvent:
    """连接池事件的结构化数据。"""

    type: PoolEventType
    timestamp: float = field(default_factory=time.time)
    conn_id: Optional[int] = None
    pool_name: str = "default"
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        conn_part = f" conn#{self.conn_id}" if self.conn_id is not None else ""
        return f"[{self.type.value}]{conn_part} {self.details}"


EventListener = Callable[[PoolEvent], None]


class EventDispatcher:
    """
    线程安全的事件分发器。

    设计要点:
    - 监听器异常完全隔离: 每个监听器调用都包 try-catch
    - 分发事件时不持锁,避免监听器阻塞其他线程
    - 支持动态添加/移除监听器
    """

    def __init__(self, pool_name: str = "default") -> None:
        self._pool_name = pool_name
        self._lock = threading.RLock()
        self._listeners: List[EventListener] = []
        self._listener_ids: dict[int, EventListener] = {}
        self._next_id = 0

    def add_listener(self, listener: EventListener) -> int:
        """
        添加一个事件监听器,返回监听器 ID(用于移除)。
        同一个函数可以被添加多次,每次返回不同 ID。
        """
        if not callable(listener):
            raise TypeError("listener must be callable")
        with self._lock:
            self._next_id += 1
            lid = self._next_id
            self._listeners.append(listener)
            self._listener_ids[lid] = listener
            return lid

    def remove_listener(self, listener_id: int) -> bool:
        """根据 ID 移除监听器,返回是否成功移除。"""
        with self._lock:
            listener = self._listener_ids.pop(listener_id, None)
            if listener is None:
                return False
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass
            return True

    def remove_all_listeners(self) -> None:
        """移除所有监听器。"""
        with self._lock:
            self._listeners.clear()
            self._listener_ids.clear()

    def dispatch(
        self,
        event_type: PoolEventType,
        conn_id: Optional[int] = None,
        **details: Any,
    ) -> None:
        """
        分发一个事件给所有监听器。

        异常隔离:
        - 每个监听器独立 try-catch
        - 单个监听器异常打 warning 日志,不影响其他监听器
        - 分发过程绝不抛出异常给调用方
        """
        # 先持锁拿到监听器快照,然后在锁外分发
        # 这样即使监听器执行时间很长,也不会阻塞 add/remove
        with self._lock:
            listeners = list(self._listeners)

        if not listeners:
            return

        event = PoolEvent(
            type=event_type,
            conn_id=conn_id,
            pool_name=self._pool_name,
            details=dict(details),
        )

        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "EventListener %s raised for event %s: %s",
                    getattr(listener, "__name__", repr(listener)),
                    event_type.value,
                    exc,
                    exc_info=True,
                )
