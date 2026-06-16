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
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, List, Optional


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

    POOL_PAUSED = "pool_paused"
    POOL_RESUMED = "pool_resumed"
    POOL_RESIZED = "pool_resized"
    HEALTH_CHECK_TRIGGERED = "health_check_triggered"


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
    线程安全的事件分发器 + 历史事件环形缓冲区。

    设计要点:
    - 监听器异常完全隔离: 每个监听器调用都包 try-catch
    - 分发事件时不持锁,避免监听器阻塞其他线程
    - 支持动态添加/移除监听器
    - 环形缓冲区保存最近 N 条事件,支持审计回放
    """

    def __init__(self, pool_name: str = "default", history_size: int = 200) -> None:
        self._pool_name = pool_name
        self._lock = threading.RLock()
        self._listeners: List[EventListener] = []
        self._listener_ids: dict[int, EventListener] = {}
        self._next_id = 0
        # 环形历史缓冲区
        self._history: Deque[PoolEvent] = deque(maxlen=history_size)
        self._history_size = history_size

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
        分发一个事件给所有监听器,并存入历史环形缓冲区。

        异常隔离:
        - 每个监听器独立 try-catch
        - 单个监听器异常打 warning 日志,不影响其他监听器
        - 分发过程绝不抛出异常给调用方
        """
        event = PoolEvent(
            type=event_type,
            conn_id=conn_id,
            pool_name=self._pool_name,
            details=dict(details),
        )

        # 存入历史环 (持锁,保证与查询历史的线程安全)
        with self._lock:
            self._history.append(event)
            listeners = list(self._listeners)

        if not listeners:
            return

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

    # ---------------------------------------------------------- 历史回放

    @property
    def history_size(self) -> int:
        """历史环形缓冲区的容量。"""
        return self._history_size

    @property
    def history_count(self) -> int:
        """当前历史缓冲区中的事件数。"""
        with self._lock:
            return len(self._history)

    def get_history(
        self,
        event_type: Optional[PoolEventType] = None,
        limit: Optional[int] = None,
    ) -> List[PoolEvent]:
        """
        查询历史事件(从新到旧)。

        :param event_type: 按事件类型过滤,None 表示返回所有类型
        :param limit: 最多返回多少条,None 表示返回全部(不超过 history_size)
        :return: 历史事件列表(最新的在前面)
        """
        with self._lock:
            events = list(reversed(self._history))
        if event_type is not None:
            events = [e for e in events if e.type == event_type]
        if limit is not None and limit > 0:
            events = events[:limit]
        return events

    def get_recent_events(self, limit: int = 10) -> List[PoolEvent]:
        """获取最近 N 条事件(从新到旧),等价于 get_history(limit=N)。"""
        return self.get_history(limit=limit)

    def clear_history(self) -> None:
        """清空历史事件缓冲区。"""
        with self._lock:
            self._history.clear()
