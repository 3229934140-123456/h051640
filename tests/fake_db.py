"""
可模拟的假数据库连接,用于单元测试。
支持模拟: 探活失败、连接关闭、执行查询等。
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class FakeCursor:
    def __init__(self) -> None:
        self.closed = False

    def execute(self, sql, *args, **kwargs):
        pass

    def fetchall(self):
        return [(1,)]

    def close(self):
        self.closed = True


class FakeDBConnection:
    """
    模拟的数据库连接。

    特性:
    - alive: 控制 ping 是否通过
    - close_count: 被调用 close() 的次数
    - rollback_count: reset 时 rollback 次数
    - operation_delay: 模拟查询耗时
    - autocommit: 模拟 DB-API 属性
    """

    _total_created = 0
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        operation_delay: float = 0.0,
        kill_after_seconds: Optional[float] = None,
    ) -> None:
        with FakeDBConnection._lock:
            FakeDBConnection._total_created += 1
            self._id = FakeDBConnection._total_created

        self.closed = False
        self.rollback_count = 0
        self.close_call_count = 0
        self.operation_delay = operation_delay
        self._created_at = time.monotonic()
        self._kill_after = kill_after_seconds
        self.autocommit = False
        self.queries_executed = 0

    @property
    def fake_id(self) -> int:
        return self._id

    @property
    def alive(self) -> bool:
        if self.closed:
            return False
        if self._kill_after is not None:
            if (time.monotonic() - self._created_at) > self._kill_after:
                return False
        return True

    # ---------------------------------------------------------- DB-API 兼容方法

    def cursor(self):
        self._check_alive()
        self.queries_executed += 1
        if self.operation_delay:
            time.sleep(self.operation_delay)
        return FakeCursor()

    def rollback(self):
        self.rollback_count += 1

    def commit(self):
        self._check_alive()

    def ping(self):
        self._check_alive()

    def close(self):
        self.close_call_count += 1
        self.closed = True

    def _check_alive(self):
        if not self.alive:
            raise ConnectionError(
                f"Fake connection #{self._id} is dead (closed={self.closed})"
            )

    def kill(self):
        """模拟被数据库单方面关闭。"""
        self.closed = True

    @classmethod
    def reset_counter(cls):
        with cls._lock:
            cls._total_created = 0
