"""
重试策略模块
============

为连接的创建、探活、重置提供统一的重试策略。
支持:
- 自定义最大重试次数
- 指数退避 (exponential backoff)
- 固定延迟
- 最大退避上限
- 可配置哪些异常需要重试 (默认所有 Exception 都重试)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type


logger = logging.getLogger("db_pool.retry")


@dataclass
class RetryPolicy:
    """
    重试策略配置。

    例子::

        # 最多重试 3 次,初始延迟 0.1s,指数退避,最大 1s
        policy = RetryPolicy(
            max_attempts=3,
            initial_delay=0.1,
            backoff_factor=2.0,
            max_delay=1.0,
            jitter=True,
        )

    重试次数 = max_attempts; 即 max_attempts=3 意味着最多尝试 4 次
    (第 1 次正常调用 + 3 次重试)。
    """

    max_attempts: int = 2
    initial_delay: float = 0.05
    backoff_factor: float = 2.0
    max_delay: float = 1.0
    jitter: bool = True
    # 重试的异常白名单; None 表示重试所有 Exception
    retry_on_exceptions: Optional[Tuple[Type[BaseException], ...]] = None
    # 每次重试回调 (attempt, delay, exc)
    on_retry: Optional[Callable[[int, float, Exception], None]] = None

    def should_retry(self, attempt: int, exc: Exception) -> bool:
        if attempt >= self.max_attempts:
            return False
        if self.retry_on_exceptions is None:
            return True
        return isinstance(exc, self.retry_on_exceptions)

    def next_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试前的等待延迟 (attempt 从 1 开始)。"""
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        if self.max_delay > 0:
            delay = min(delay, self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random())  # 0.5 ~ 1.5 倍
        return max(0.0, delay)


class RetryOutcome:
    """一次带重试操作的结果封装。"""

    __slots__ = ("ok", "value", "last_exc", "attempts", "total_delay")

    def __init__(
        self,
        ok: bool,
        value: Any = None,
        last_exc: Optional[Exception] = None,
        attempts: int = 0,
        total_delay: float = 0.0,
    ) -> None:
        self.ok = ok
        self.value = value
        self.last_exc = last_exc
        self.attempts = attempts
        self.total_delay = total_delay

    def __bool__(self) -> bool:
        return self.ok


def run_with_retry(
    func: Callable[[], Any],
    policy: RetryPolicy,
) -> RetryOutcome:
    """
    执行 func,按 policy 重试。返回 RetryOutcome。

    统计:
    - attempts: 实际重试次数 (0 表示一次成功)
    - total_delay: 重试期间累计 sleep 秒数
    """
    last_exc: Optional[Exception] = None
    attempts = 0
    total_delay = 0.0

    # 第 0 次: 初始尝试
    try:
        result = func()
        return RetryOutcome(ok=True, value=result, last_exc=None,
                            attempts=0, total_delay=0.0)
    except Exception as exc:  # noqa: BLE001
        last_exc = exc

    # 重试循环
    while policy.should_retry(attempts, last_exc):  # type: ignore[arg-type]
        delay = policy.next_delay(attempts + 1)
        total_delay += delay

        if policy.on_retry is not None:
            try:
                policy.on_retry(attempts + 1, delay, last_exc)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                pass

        logger.debug(
            "Retrying %s (attempt %d/%d, delay %.3fs): %s",
            getattr(func, "__name__", "func"),
            attempts + 1, policy.max_attempts, delay, last_exc,
        )

        if delay > 0:
            time.sleep(delay)

        attempts += 1
        try:
            result = func()
            return RetryOutcome(ok=True, value=result, last_exc=last_exc,
                                attempts=attempts, total_delay=total_delay)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    # 所有重试均失败
    return RetryOutcome(ok=False, value=None, last_exc=last_exc,
                        attempts=attempts, total_delay=total_delay)


# 默认无重试的策略,方便作为默认值
NO_RETRY = RetryPolicy(max_attempts=0, initial_delay=0.0)
