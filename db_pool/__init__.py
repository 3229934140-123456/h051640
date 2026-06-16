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
"""

from .pool import ConnectionPool, PoolConfig
from .connection import PooledConnection

__all__ = ["ConnectionPool", "PoolConfig", "PooledConnection"]
__version__ = "1.0.0"
