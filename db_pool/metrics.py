"""
指标导出模块
============

提供可观测性指标导出:
- get_stats_dict(): 返回 dict 形式,方便日志采集
- get_stats_json(): 返回 JSON 字符串
- get_prometheus_metrics(): 返回 Prometheus exposition format

所有计数器都是 monotonic increasing 的,gauges 是当前瞬时值。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from .pool_manager import PoolStats


METRIC_PREFIX = "db_pool_"


def stats_to_dict(
    stats: PoolStats,
    pool_name: str = "default",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """转换为适合日志打印的 dict。"""
    d = {
        "pool_name": pool_name,
        "timestamp": time.time(),
        "config": {
            "min_size": stats.min_size,
            "max_size": stats.max_size,
        },
        "gauge": {
            "total": stats.total,
            "idle": stats.idle,
            "borrowed": stats.borrowed,
            "waiting": stats.waiting,
            "pending_creates": getattr(stats, "pending_creates", 0),
            "in_health_check": getattr(stats, "in_health_check", 0),
        },
        "counter": {
            "created": stats.created,
            "destroyed": stats.destroyed,
            "borrowed_total": stats.borrowed_total,
            "timeouts": stats.timeouts,
            "leaked": stats.leaked,
            "create_failures": getattr(stats, "create_failures", 0),
            "ping_failures": getattr(stats, "ping_failures", 0),
            "reset_failures": getattr(stats, "reset_failures", 0),
            "retried_operations": getattr(stats, "retried_operations", 0),
            "rotated_total": getattr(stats, "rotated_total", 0),
        },
        "timing": {
            "avg_wait_seconds": round(getattr(stats, "avg_wait_seconds", 0.0), 6),
            "p99_wait_seconds": round(getattr(stats, "p99_wait_seconds", 0.0), 6),
            "max_wait_seconds": round(getattr(stats, "max_wait_seconds", 0.0), 6),
        },
        "last_events": {
            "last_create_reason": getattr(stats, "last_create_reason", ""),
            "last_destroy_reason": getattr(stats, "last_destroy_reason", ""),
            "last_rotation_reason": getattr(stats, "last_rotation_reason", ""),
        },
    }
    if extra:
        d.update(extra)
    return d


def stats_to_json(
    stats: PoolStats,
    pool_name: str = "default",
    extra: Optional[Dict[str, Any]] = None,
    indent: Optional[int] = 2,
) -> str:
    return json.dumps(
        stats_to_dict(stats, pool_name=pool_name, extra=extra),
        indent=indent,
        default=str,
    )


def stats_to_prometheus(
    stats: PoolStats,
    pool_name: str = "default",
) -> str:
    """
    导出为 Prometheus 文本格式。
    可直接挂在 /metrics 端点后面返回。
    """
    lines: list[str] = []
    now_ms = int(time.time() * 1000)
    labels = f'pool_name="{pool_name}"'

    def _gauge(name: str, value: int | float, help_text: str) -> None:
        lines.append(f"# HELP {METRIC_PREFIX}{name} {help_text}")
        lines.append(f"# TYPE {METRIC_PREFIX}{name} gauge")
        lines.append(f"{METRIC_PREFIX}{name}{{{labels}}} {value} {now_ms}")

    def _counter(name: str, value: int | float, help_text: str) -> None:
        lines.append(f"# HELP {METRIC_PREFIX}{name} {help_text}")
        lines.append(f"# TYPE {METRIC_PREFIX}{name} counter")
        lines.append(f"{METRIC_PREFIX}{name}{{{labels}}} {value} {now_ms}")

    _gauge("min_size", stats.min_size, "Minimum pool size")
    _gauge("max_size", stats.max_size, "Maximum pool size")
    _gauge("total", stats.total, "Current total connections")
    _gauge("idle", stats.idle, "Current idle connections")
    _gauge("borrowed", stats.borrowed, "Current borrowed connections")
    _gauge("waiting", stats.waiting, "Current threads waiting for a connection")
    _gauge(
        "pending_creates",
        getattr(stats, "pending_creates", 0),
        "Connections currently being created",
    )
    _gauge(
        "in_health_check",
        getattr(stats, "in_health_check", 0),
        "Connections currently in health check (moved from idle)",
    )

    _counter("created_total", stats.created, "Total connections created")
    _counter("destroyed_total", stats.destroyed, "Total connections destroyed")
    _counter("borrowed_total", stats.borrowed_total, "Total borrow operations")
    _counter("timeouts_total", stats.timeouts, "Total borrow timeouts")
    _counter("leaked_total", stats.leaked, "Total leaked connections detected")
    _counter(
        "create_failures_total",
        getattr(stats, "create_failures", 0),
        "Total connection create failures",
    )
    _counter(
        "ping_failures_total",
        getattr(stats, "ping_failures", 0),
        "Total ping failures on borrow or health check",
    )
    _counter(
        "reset_failures_total",
        getattr(stats, "reset_failures", 0),
        "Total reset failures on return",
    )
    _counter(
        "retried_operations_total",
        getattr(stats, "retried_operations", 0),
        "Total retried operations (create/ping/reset)",
    )
    _counter(
        "rotated_total",
        getattr(stats, "rotated_total", 0),
        "Total connections rotated due to max_borrow_count or max_age_for_rotation",
    )

    # Timing 作为 gauge 输出(滑动窗口统计)
    _gauge(
        "avg_wait_seconds",
        round(getattr(stats, "avg_wait_seconds", 0.0), 6),
        "Average borrow wait time in seconds (sliding window)",
    )
    _gauge(
        "p99_wait_seconds",
        round(getattr(stats, "p99_wait_seconds", 0.0), 6),
        "P99 borrow wait time in seconds (sliding window)",
    )
    _gauge(
        "max_wait_seconds",
        round(getattr(stats, "max_wait_seconds", 0.0), 6),
        "Max borrow wait time in seconds (sliding window)",
    )

    return "\n".join(lines) + "\n"
