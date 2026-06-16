"""
滑动窗口统计工具
==============

高效统计最近 N 秒 / 最近 M 个样本的 avg / p99 / max。
用于:
- 借出等待耗时 (acquire wait time)
- 借出时如果容量小数据量场景，环形缓冲区 + 懒清理过期样本。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class _Sample:
    value: float
    timestamp: float


class TimeWindowStats:
    """
    滑动窗口统计器。线程安全。

    :param window_seconds: 统计最近多少秒内的样本
    :param max_samples: 最多保留多少个样本（防止内存无限增长）
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_samples: int = 2000,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")

        self._window = window_seconds
        self._max_samples = max_samples
        self._lock = threading.RLock()
        self._samples: Deque[_Sample] = deque()
        self._max: float = 0.0

    # ---------------------------------------------------------- 记录

    def record(self, value: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append(_Sample(value=value, timestamp=now))
            if value > self._max:
                self._max = value
            # 超量裁剪最老的
            while len(self._samples) > self._max_samples:
                self._samples.popleft()

    # ---------------------------------------------------------- 统计

    def _prune(self) -> List[float]:
        """清理过期样本,返回未过期的 value 列表。"""
        now = time.monotonic()
        cutoff = now - self._window
        values: List[float] = []
        # 从前往后找,丢掉过期的
        with self._lock:
            while self._samples and self._samples[0].timestamp < cutoff:
                self._samples.popleft()
            # 重算 max
            new_max = 0.0
            for s in self._samples:
                values.append(s.value)
                if s.value > new_max:
                    new_max = s.value
            self._max = new_max
            return values

    def avg(self) -> float:
        values = self._prune()
        if not values:
            return 0.0
        return sum(values) / len(values)

    def p99(self) -> float:
        values = sorted(self._prune())
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        # 99 百分位,线性插值
        idx = int(len(values) * 0.99)
        if idx >= len(values):
            idx = len(values) - 1
        return values[idx]

    def max(self) -> float:
        self._prune()
        with self._lock:
            return self._max

    def count(self) -> int:
        self._prune()
        with self._lock:
            return len(self._samples)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._max = 0.0
