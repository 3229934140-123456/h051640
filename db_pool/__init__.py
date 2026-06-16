"""
数据库连接池库
============

模块构成:
- connection_factory: 连接工厂,负责连接的创建与销毁
- pool_manager: 池状态管理,连接数控制与动态伸缩
- borrow_return: 借还逻辑,等待队列与超时机制
- health_check: 健康检查,空闲连接探活
- leak_detector: 泄漏检测,长时间未归还连接告警
- pool: 对外主入口 ConnectionPool
- retry: 重试策略 (指数退避 + 抖动)
- shutdown_state: 关闭状态机 (可观测停机流程)
- metrics: 指标导出 (dict/JSON/Prometheus)
- time_window_stats: 滑动窗口统计 (avg/p99/max 耗时)
"""

from .pool import ConnectionPool, PoolConfig
from .connection import PooledConnection, ConnectionReturnedError
from .connection_factory import ConnectionFactory
from .pool_manager import PoolStats
from .borrow_return import GetTimeoutError, PoolClosedError, PoolPausedError
from .leak_detector import LeakInfo, LeakListener
from .retry import RetryPolicy, RetryOutcome, NO_RETRY
from .shutdown_state import ShutdownInfo, ShutdownPhase, ShutdownState
from .metrics import stats_to_dict, stats_to_json, stats_to_prometheus
from .events import EventDispatcher, PoolEvent, PoolEventType, EventListener

__all__ = [
    "ConnectionPool",
    "PoolConfig",
    "PooledConnection",
    "ConnectionFactory",
    "PoolStats",
    "GetTimeoutError",
    "PoolClosedError",
    "PoolPausedError",
    "ConnectionReturnedError",
    "LeakInfo",
    "LeakListener",
    "RetryPolicy",
    "RetryOutcome",
    "NO_RETRY",
    "ShutdownInfo",
    "ShutdownPhase",
    "ShutdownState",
    "EventDispatcher",
    "PoolEvent",
    "PoolEventType",
    "EventListener",
    "stats_to_dict",
    "stats_to_json",
    "stats_to_prometheus",
]
__version__ = "2.2.0"
