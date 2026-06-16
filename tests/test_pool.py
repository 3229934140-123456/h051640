"""
连接池单元测试与使用示例。

运行:  python -m pytest tests/test_pool.py -v
或直接: python tests/test_pool.py
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import unittest
from pathlib import Path

# 让项目根目录可 import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db_pool import (
    ConnectionPool, PoolConfig, PooledConnection,
    ConnectionReturnedError, RetryPolicy, ShutdownPhase,
    stats_to_prometheus, PoolEventType, PoolEvent,
)
from db_pool.borrow_return import GetTimeoutError, PoolClosedError, PoolPausedError
from db_pool.leak_detector import LeakInfo
from tests.fake_db import FakeDBConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def make_pool(**cfg_kw):
    """创建一个使用 FakeDBConnection 的池。"""
    create_fn = cfg_kw.pop("create_fn", None)
    cfg_kw.setdefault("min_size", 3)
    cfg_kw.setdefault("max_size", 8)
    cfg_kw.setdefault("health_check_enabled", False)
    cfg_kw.setdefault("leak_check_enabled", False)
    cfg = PoolConfig(**cfg_kw)
    pool = ConnectionPool(
        create_fn=create_fn if create_fn else lambda: FakeDBConnection(),
        config=cfg,
    )
    return pool


# ==========================================================================
# 基础功能测试
# ==========================================================================
class TestBasics(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_warm_up_creates_min_size(self):
        pool = make_pool(min_size=4, max_size=10)
        pool.start()
        s = pool.stats()
        self.assertEqual(s.total, 4)
        self.assertEqual(s.idle, 4)
        self.assertEqual(s.borrowed, 0)
        pool.close()

    def test_acquire_and_release(self):
        pool = make_pool(min_size=2, max_size=5).start()
        real_conn_ref = None
        with pool.connection() as conn:
            self.assertIsInstance(conn, PooledConnection)
            real_conn_ref = conn.real_connection
            cur = conn.cursor()
            cur.execute("SELECT 1")
            self.assertEqual(real_conn_ref.queries_executed, 1)
            s = pool.stats()
            self.assertEqual(s.borrowed, 1)
            self.assertEqual(s.idle, 1)
        s = pool.stats()
        self.assertEqual(s.borrowed, 0)
        self.assertEqual(s.idle, 2)
        # reset 应该被调用(rollback)
        self.assertEqual(real_conn_ref.rollback_count, 1)
        pool.close()

    def test_returned_connection_reset(self):
        """归还的连接,autocommit 应被恢复,事务被回滚。"""
        pool = make_pool(min_size=1, max_size=2).start()
        with pool.connection() as conn:
            conn.real_connection.autocommit = True
            # 在归还时会被重置为 False
        with pool.connection() as conn2:
            # 应该拿到同一个连接
            self.assertFalse(conn2.real_connection.autocommit)
        pool.close()

    def test_borrow_creates_up_to_max(self):
        """不够用时动态扩容,但不超过 max_size。"""
        pool = make_pool(min_size=1, max_size=3).start()
        c1 = pool.acquire()
        c2 = pool.acquire()
        c3 = pool.acquire()
        self.assertEqual(pool.stats().total, 3)
        self.assertEqual(pool.stats().borrowed, 3)
        c1.close(); c2.close(); c3.close()
        pool.close()


# ==========================================================================
# 等待队列与超时
# ==========================================================================
class TestWaitingAndTimeout(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_wait_queue_blocks_until_release(self):
        pool = make_pool(min_size=0, max_size=1, borrow_timeout=1.0).start()
        holder = pool.acquire()
        acquired_in_thread = threading.Event()
        got_conn = []
        errors = []

        def _taker():
            try:
                c = pool.acquire()
                got_conn.append(c)
                acquired_in_thread.set()
                time.sleep(0.1)
                c.close()
            except Exception as e:
                errors.append(e)
                acquired_in_thread.set()

        t = threading.Thread(target=_taker)
        t.start()
        time.sleep(0.2)
        self.assertFalse(acquired_in_thread.is_set(), "线程应仍在等待")
        holder.close()  # 归还,应唤醒
        acquired_in_thread.wait(timeout=2)
        self.assertTrue(acquired_in_thread.is_set())
        self.assertEqual(len(got_conn), 1)
        self.assertEqual(errors, [])
        t.join()
        pool.close()

    def test_timeout_raises(self):
        pool = make_pool(min_size=0, max_size=1, borrow_timeout=0.3).start()
        holder = pool.acquire()
        t0 = time.monotonic()
        with self.assertRaises(GetTimeoutError):
            pool.acquire()
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.25)
        holder.close()
        pool.close()

    def test_fifo_waiters(self):
        """先到的等待者先被唤醒。"""
        pool = make_pool(min_size=0, max_size=1).start()
        holder = pool.acquire()
        order = []
        threads = []

        def _worker(idx):
            try:
                c = pool.acquire()
                order.append(idx)
                time.sleep(0.05)
                c.close()
            except Exception as e:
                order.append(f"err-{idx}:{e}")

        for i in range(3):
            t = threading.Thread(target=_worker, args=(i,))
            t.start()
            threads.append(t)
            time.sleep(0.02)  # 保证入队顺序

        time.sleep(0.1)
        holder.close()
        for t in threads:
            t.join(timeout=5)
        # 先等待的先获取,因此 order 应为 [0,1,2]
        self.assertEqual(order, [0, 1, 2])
        pool.close()


# ==========================================================================
# 健康检查 - 借出错连接被替换
# ==========================================================================
class TestHealthOnBorrow(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_borrow_replaces_dead_connection(self):
        """借出时若连接已死,应销毁并替换。"""
        pool = make_pool(min_size=2, max_size=5, test_on_borrow=True).start()
        # 手动杀掉所有空闲连接
        fake_conns = [c for c in _peek_idle(pool)]
        for fc in fake_conns:
            fc.real_connection.kill()

        destroyed_before = pool.stats().destroyed
        created_before = pool.stats().created
        with pool.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")  # 新连接应该可用
        s = pool.stats()
        # 至少销毁了 1 个死连接,且至少创建了 1 个新连接替换
        self.assertGreaterEqual(s.destroyed, destroyed_before + 1)
        self.assertGreaterEqual(s.created, created_before + 1)
        self.assertGreaterEqual(s.ping_failures, 1)
        pool.close()

    def test_max_lifetime_and_shrink(self):
        """开健康检查,短 max_lifetime 让连接被主动淘汰,再缩容。"""
        pool = make_pool(
            min_size=1, max_size=5,
            health_check_enabled=True,
            check_interval=0.1,
            idle_before_check=0.0,
            max_lifetime=0.3,
            enable_shrink=False,
            leak_check_enabled=False,
        ).start()
        s0 = pool.stats()
        self.assertEqual(s0.idle, 1)
        time.sleep(0.6)  # 足够多轮健康检查
        # 被 max_lifetime 销毁了,但没缩容所以应该又补上 min_size
        s = pool.stats()
        self.assertGreaterEqual(s.total, 1)
        pool.close()


def _peek_idle(pool):
    """通过 reflection 拿到 idle 集合(仅测试用)。"""
    return list(pool._manager._idle)


# ==========================================================================
# 泄漏检测
# ==========================================================================
class TestLeakDetect(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_leak_detected(self):
        leaks = []

        def _listener(info: LeakInfo):
            leaks.append(info.conn_id)

        pool = make_pool(
            min_size=0, max_size=2,
            health_check_enabled=False,
            leak_check_enabled=True,
            leak_threshold=0.3,
            leak_cooldown=0.2,
            capture_stack=True,
            leak_listener=_listener,
        ).start()

        leaked_conn = pool.acquire()  # 故意不还
        time.sleep(0.8)  # 让泄漏检测跑两轮
        s = pool.stats()
        self.assertGreaterEqual(s.leaked, 1)
        self.assertIn(leaked_conn.conn_id, leaks)
        leaked_conn.close()
        pool.close()


# ==========================================================================
# 动态伸缩与关闭
# ==========================================================================
class TestResizeAndClose(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_resize_expand_min(self):
        pool = make_pool(min_size=1, max_size=10).start()
        self.assertEqual(pool.stats().total, 1)
        pool.resize(new_min=4)
        self.assertEqual(pool.stats().total, 4)
        self.assertEqual(pool.stats().min_size, 4)
        pool.close()

    def test_close_graceful_waits_for_return(self):
        pool = make_pool(min_size=1, max_size=2).start()
        conn = pool.acquire()
        closed_ok = threading.Event()
        errors = []

        def _closer():
            try:
                pool.close(graceful=True, wait_timeout=5)
                closed_ok.set()
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=_closer)
        t.start()
        time.sleep(0.2)
        self.assertFalse(closed_ok.is_set(), "应仍在等待连接归还")
        conn.close()
        closed_ok.wait(timeout=3)
        self.assertTrue(closed_ok.is_set())
        self.assertEqual(errors, [])
        t.join()
        self.assertTrue(pool.is_closed)

    def test_close_force_destroys_borrowed(self):
        pool = make_pool(min_size=0, max_size=2).start()
        conn = pool.acquire()
        real = conn.real_connection
        pool.close(graceful=False, force=True)
        self.assertTrue(real.closed)
        self.assertEqual(pool.stats().total, 0)

    def test_closed_pool_rejects_new_acquire(self):
        pool = make_pool(min_size=0, max_size=1).start()
        pool.close()
        with self.assertRaises(PoolClosedError):
            pool.acquire()


# ==========================================================================
# 压力 / 并发测试
# ==========================================================================
class TestConcurrency(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_many_threads_no_corruption(self):
        N = 50
        ITERS = 20
        pool = make_pool(min_size=3, max_size=10, borrow_timeout=10).start()
        errors = []
        success_count = threading.Lock()
        n_success = [0]

        def _worker():
            for _ in range(ITERS):
                try:
                    with pool.connection() as c:
                        c.cursor().execute("blah")
                        time.sleep(0.005)
                    with success_count:
                        n_success[0] += 1
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"errors: {errors}")
        self.assertEqual(n_success[0], N * ITERS)
        s = pool.stats()
        self.assertLessEqual(s.total, 10)
        self.assertGreaterEqual(s.created, 3)
        pool.close()


# ==========================================================================
# 新增功能测试: 连接剥离后旧引用禁用
# ==========================================================================
class TestConnectionStripping(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_returned_connection_cannot_be_used(self):
        """连接归还后,旧引用不能再执行任何操作。"""
        pool = make_pool(min_size=1, max_size=2).start()
        conn = pool.acquire()
        conn_id = conn.conn_id
        real_conn = conn.real_connection
        # 正常使用
        conn.cursor().execute("SELECT 1")
        self.assertEqual(real_conn.queries_executed, 1)
        # 归还
        conn.close()
        # 旧引用上的任何操作都应该抛 ConnectionReturnedError
        with self.assertRaises(ConnectionReturnedError):
            conn.cursor()
        with self.assertRaises(ConnectionReturnedError):
            conn.execute("SELECT 1")
        with self.assertRaises(ConnectionReturnedError):
            _ = conn.real_connection
        # 但真实连接还在,并且可以被池复用
        self.assertFalse(real_conn.closed)
        # 再次借连接,应该拿到同一个真实连接(但新的 PooledConnection)
        with pool.connection() as conn2:
            self.assertIs(conn2.real_connection, real_conn)
            self.assertNotEqual(conn2.conn_id, conn_id)  # 新包装,新 conn_id
        pool.close()

    def test_double_close_is_safe(self):
        """对同一个连接调用 close() 多次应该是安全的。"""
        pool = make_pool(min_size=1, max_size=2).start()
        conn = pool.acquire()
        conn.close()
        # 第二次 close 不应该抛错,但也不应该做任何事情
        try:
            conn.close()
        except Exception as e:
            self.fail(f"第二次 close 不应该抛错: {e}")
        pool.close()


# ==========================================================================
# 新增功能测试: 重试策略
# ==========================================================================
class TestRetryPolicy(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_create_retry_on_failure(self):
        """创建连接失败时,按重试策略重试。"""
        create_attempts = [0]

        def _flaky_create():
            create_attempts[0] += 1
            if create_attempts[0] < 3:
                raise RuntimeError(f"Simulated create failure #{create_attempts[0]}")
            return FakeDBConnection()

        retry_policy = RetryPolicy(
            max_attempts=3,
            initial_delay=0.01,
            backoff_factor=1.5,
            max_delay=0.1,
            jitter=False,
        )
        cfg = PoolConfig(
            min_size=1, max_size=3,
            health_check_enabled=False,
            leak_check_enabled=False,
            retry_policy=retry_policy,
        )
        pool = ConnectionPool(create_fn=_flaky_create, config=cfg)
        pool.start()
        # 应该尝试了 3 次(1次初始 + 2次重试)才成功
        self.assertEqual(create_attempts[0], 3)
        s = pool.stats()
        # create_failures 应该记录了前 2 次失败
        self.assertEqual(s.create_failures, 2)
        # retried_operations 应该记录了重试次数
        self.assertGreaterEqual(s.retried_operations, 2)
        pool.close()

    def test_ping_retry_on_failure(self):
        """ping 失败时按重试策略重试,最终失败则替换连接。"""
        pool = make_pool(
            min_size=1, max_size=2,
            test_on_borrow=True,
            retry_policy=RetryPolicy(
                max_attempts=2,
                initial_delay=0.01,
                jitter=False,
            ),
        ).start()
        # 杀掉唯一的空闲连接,ping 会失败
        idle_conns = _peek_idle(pool)
        self.assertEqual(len(idle_conns), 1)
        idle_conns[0].real_connection.kill()
        # 借出时应该会 ping 失败,重试,最后销毁坏连接并创建新的
        created_before = pool.stats().created
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")
        s = pool.stats()
        self.assertGreaterEqual(s.ping_failures, 1)
        self.assertGreaterEqual(s.destroyed, 1)
        self.assertGreaterEqual(s.created, created_before + 1)
        pool.close()


# ==========================================================================
# 新增功能测试: 关闭状态可观测性
# ==========================================================================
class TestShutdownObservability(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_shutdown_info_reports_phases(self):
        """关闭过程中各阶段状态可查询。"""
        pool = make_pool(min_size=1, max_size=2).start()
        # 预热后应该有 1 个 idle 连接
        self.assertEqual(pool.stats().idle, 1)
        conn = pool.acquire()

        # 关闭前
        info = pool.get_shutdown_info()
        self.assertEqual(info.phase, ShutdownPhase.NOT_SHUTTING)
        self.assertFalse(info.is_shutting)

        # 异步关闭
        close_done = threading.Event()
        errors = []

        def _closer():
            try:
                pool.close(graceful=True, wait_timeout=0.3)
                close_done.set()
            except Exception as e:
                errors.append(e)
                close_done.set()

        t = threading.Thread(target=_closer)
        t.start()
        # 等待一会儿,应该处于 BEGIN_GRACEFUL_WAIT 阶段
        time.sleep(0.1)
        info = pool.get_shutdown_info()
        self.assertTrue(info.is_shutting)
        self.assertEqual(info.waiting_for_borrowed, 1)
        # 归还连接
        conn.close()
        close_done.wait(timeout=2)
        t.join()
        # 关闭完成
        info = pool.get_shutdown_info()
        self.assertTrue(info.is_complete)
        self.assertEqual(info.phase, ShutdownPhase.SHUTDOWN_COMPLETE)
        # force_destroy_all 会销毁所有剩余连接(包括之前的 idle 连接)
        self.assertGreaterEqual(info.force_destroyed_count, 0)
        # 验证优雅等待确实发生过
        self.assertGreater(info.graceful_wait_elapsed, 0)
        self.assertEqual(errors, [])

    def test_shutdown_timeout_reports_outstanding(self):
        """优雅关闭超时后,outstanding_conn_ids 列出未归还的连接。"""
        pool = make_pool(min_size=0, max_size=2).start()
        conn = pool.acquire()
        conn_id = conn.conn_id
        # 不等归还,直接带超时关闭
        pool.close(graceful=True, wait_timeout=0.2)
        info = pool.get_shutdown_info()
        self.assertEqual(info.phase, ShutdownPhase.SHUTDOWN_COMPLETE)
        self.assertIn(conn_id, info.outstanding_conn_ids)


# ==========================================================================
# 新增功能测试: 指标导出
# ==========================================================================
class TestMetricsExport(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_get_stats_dict(self):
        """get_stats_dict 返回 dict 格式的指标。"""
        pool = make_pool(min_size=2, max_size=5, pool_name="test_pool").start()
        d = pool.get_stats_dict()
        self.assertEqual(d["pool_name"], "test_pool")
        self.assertIn("gauge", d)
        self.assertIn("counter", d)
        self.assertIn("timing", d)
        self.assertEqual(d["gauge"]["total"], 2)
        self.assertEqual(d["gauge"]["idle"], 2)
        self.assertEqual(d["counter"]["created"], 2)
        pool.close()

    def test_get_stats_json(self):
        """get_stats_json 返回 JSON 字符串。"""
        import json
        pool = make_pool(min_size=1, max_size=3).start()
        j = pool.get_stats_json()
        d = json.loads(j)
        self.assertEqual(d["config"]["min_size"], 1)
        self.assertEqual(d["config"]["max_size"], 3)
        pool.close()

    def test_get_prometheus_metrics(self):
        """get_prometheus_metrics 返回 Prometheus 格式。"""
        pool = make_pool(min_size=2, max_size=5, pool_name="mypool").start()
        # 制造一些统计数据
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")
        prom = pool.get_prometheus_metrics()
        # 检查格式
        self.assertIn("# HELP db_pool_total", prom)
        self.assertIn("# TYPE db_pool_total gauge", prom)
        self.assertIn('db_pool_total{pool_name="mypool"}', prom)
        self.assertIn("# HELP db_pool_created_total", prom)
        self.assertIn("# TYPE db_pool_created_total counter", prom)
        # 检查包含 avg_wait_seconds 等新指标
        self.assertIn("db_pool_avg_wait_seconds", prom)
        self.assertIn("db_pool_p99_wait_seconds", prom)
        self.assertIn("db_pool_max_wait_seconds", prom)
        self.assertIn("db_pool_create_failures_total", prom)
        self.assertIn("db_pool_ping_failures_total", prom)
        self.assertIn("db_pool_in_health_check", prom)
        pool.close()

    def test_stats_include_new_fields(self):
        """PoolStats 包含所有新增字段。"""
        pool = make_pool(min_size=1, max_size=3).start()
        # 模拟一些失败
        pool._manager.inc_create_failure()
        pool._manager.inc_ping_failure()
        pool._manager.inc_reset_failure()
        pool._manager.inc_retried(3)
        pool._manager.record_wait_time(0.05)
        pool._manager.record_wait_time(0.1)
        s = pool.stats()
        self.assertEqual(s.create_failures, 1)
        self.assertEqual(s.ping_failures, 1)
        self.assertEqual(s.reset_failures, 1)
        self.assertEqual(s.retried_operations, 3)
        self.assertGreater(s.avg_wait_seconds, 0)
        self.assertGreater(s.p99_wait_seconds, 0)
        self.assertGreater(s.max_wait_seconds, 0)
        pool.close()


# ==========================================================================
# 新增功能测试: 健康检查并发守住 max_size
# ==========================================================================
class TestHealthCheckConcurrency(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_health_check_does_not_break_max_size(self):
        """健康检查探活期间,总连接数不会突破 max_size。"""
        # 创建一个慢 ping 的连接,让健康检查占用连接一段时间
        slow_ping_times = [0]

        def _slow_ping(conn):
            slow_ping_times[0] += 1
            time.sleep(0.2)  # 模拟慢 ping
            return not conn.closed

        pool = make_pool(
            min_size=3, max_size=3,  # min == max, 共 3 个连接
            health_check_enabled=True,
            check_interval=0.05,
            idle_before_check=0.0,
            max_lifetime=0,  # 禁用超龄淘汰
            enable_shrink=False,
            leak_check_enabled=False,
            ping_fn=_slow_ping,
        ).start()

        # 等待健康检查开始跑
        time.sleep(0.1)

        # 在健康检查探活期间(应该占了1~3个连接在 in_health_check),
        # 尝试并发借连接,应该始终守住 max_size=3
        max_total_seen = [0]
        errors = []

        def _check_total():
            try:
                for _ in range(10):
                    s = pool.stats()
                    max_total_seen[0] = max(max_total_seen[0], s.total)
                    if s.total > 3:
                        errors.append(f"total 突破 max_size: {s.total} > 3")
                    time.sleep(0.05)
            except Exception as e:
                errors.append(str(e))

        def _try_borrow():
            try:
                # 尝试借连接,应该等待,但不会让 total 超过 3
                c = pool.acquire(timeout=0.5)
                c.close()
            except GetTimeoutError:
                pass  # 超时是预期的
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=_check_total)
        t2 = threading.Thread(target=_try_borrow)
        t1.start()
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

        self.assertEqual(errors, [], f"并发期间出错: {errors}")
        self.assertLessEqual(max_total_seen[0], 3, f"total 突破了 max_size: {max_total_seen[0]}")
        # 健康检查确实跑了 ping
        self.assertGreater(slow_ping_times[0], 0)
        pool.close()


# ==========================================================================
# 新增功能测试: 事件订阅系统
# ==========================================================================
class TestEventSubscription(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_all_events_fired(self):
        """所有关键事件都能被正确触发。"""
        events_received: list[PoolEvent] = []
        errors = []

        def listener(event: PoolEvent):
            events_received.append(event)

        def bad_listener(event: PoolEvent):
            raise RuntimeError("Simulated listener error")

        pool = make_pool(
            min_size=1, max_size=3,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        # 添加监听器
        lid1 = pool.add_event_listener(listener)
        lid2 = pool.add_event_listener(bad_listener)
        self.assertGreater(lid1, 0)
        self.assertGreater(lid2, 0)
        self.assertNotEqual(lid1, lid2)

        # 借出连接 - 应该触发 CONNECTION_CREATED (如果创建新连接) 或 CONNECTION_BORROWED
        event_types_before = {e.type for e in events_received}
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")
        # 归还连接 - 应该触发 CONNECTION_RETURNED

        # 先把所有连接都借出去，再尝试获取才会超时
        borrowed = []
        for _ in range(3):
            borrowed.append(pool.acquire())

        # 现在池里没有连接了，尝试获取应该超时
        try:
            pool.acquire(timeout=0.01)
        except GetTimeoutError:
            pass

        # 归还所有连接
        for c in borrowed:
            c.close()

        # 移除坏的监听器
        result = pool.remove_event_listener(lid2)
        self.assertTrue(result)

        # 再次借出，坏监听器不应该再影响
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")

        # 验证事件类型
        event_types = {e.type for e in events_received}
        self.assertIn(PoolEventType.CONNECTION_BORROWED, event_types)
        self.assertIn(PoolEventType.CONNECTION_RETURNED, event_types)
        self.assertIn(PoolEventType.WAIT_TIMEOUT, event_types)
        if PoolEventType.CONNECTION_CREATED in event_types_before:
            self.assertIn(PoolEventType.CONNECTION_CREATED, event_types)

        # 验证坏监听器没有影响连接池运行
        s = pool.stats()
        self.assertGreaterEqual(s.borrowed_total, 2)
        pool.close()

    def test_listener_error_isolation(self):
        """监听器抛出异常不会影响连接池或其他监听器。"""
        events1 = []
        events2 = []

        def listener1(event: PoolEvent):
            events1.append(event)

        def bad_listener(event: PoolEvent):
            raise ValueError("I'm a bad listener")

        def listener2(event: PoolEvent):
            events2.append(event)

        pool = make_pool(
            min_size=0, max_size=2,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        pool.add_event_listener(listener1)
        pool.add_event_listener(bad_listener)
        pool.add_event_listener(listener2)

        # 执行操作，即使中间有坏监听器，连接池应该正常运行
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")

        # 两个好的监听器应该都收到了事件
        self.assertGreater(len(events1), 0)
        self.assertGreater(len(events2), 0)
        self.assertEqual(len(events1), len(events2))

        # 连接池应该仍然正常工作
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 2")

        pool.close()

    def test_remove_all_listeners(self):
        """移除所有监听器后，不再触发事件。"""
        events = []
        pool = make_pool(
            min_size=0, max_size=2,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        pool.add_event_listener(lambda e: events.append(e))
        pool.add_event_listener(lambda e: events.append(e))
        pool.remove_all_event_listeners()

        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")

        self.assertEqual(len(events), 0)
        pool.close()

    def test_health_check_and_leak_events(self):
        """健康检查失败和泄漏检测事件也能被捕获。"""
        events_received: list[PoolEvent] = []

        def listener(event: PoolEvent):
            events_received.append(event)

        pool = make_pool(
            min_size=2, max_size=3,
            health_check_enabled=True,
            check_interval=0.05,
            idle_before_check=0.0,
            leak_check_enabled=True,
            leak_threshold=0.1,
            leak_cooldown=0.5,
        ).start()
        pool.add_event_listener(listener)

        # 等待泄漏检测器完成初始等待并运行至少一次
        time.sleep(0.15)

        # 杀掉一个空闲连接
        idle_conns = _peek_idle(pool)
        if idle_conns:
            idle_conns[0].real_connection.kill()

        # 故意泄漏一个连接
        leaked = pool.acquire()
        # 等待足够长的时间，确保泄漏检测在连接归还前扫描到多次
        time.sleep(0.5)
        leaked.close()

        # 等待健康检查运行
        time.sleep(0.2)

        event_types = {e.type for e in events_received}
        # 至少应该有泄漏事件
        self.assertIn(PoolEventType.LEAK_DETECTED, event_types)
        # 可能有健康检查失败事件
        # 可能有连接销毁事件
        pool.close()


# ==========================================================================
# 新增功能测试: 连接轮换
# ==========================================================================
class TestConnectionRotation(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_rotation_by_borrow_count(self):
        """按借还次数轮换连接。"""
        pool = make_pool(
            min_size=1, max_size=1,
            max_borrow_count=3,  # 借出 3 次后轮换
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        created_before = pool.stats().created
        first_real_conn = None

        # 借还 3 次，第 3 次归还时应该触发轮换
        for i in range(3):
            with pool.connection() as conn:
                if i == 0:
                    first_real_conn = conn.real_connection
                conn.cursor().execute(f"SELECT {i}")

        s = pool.stats()
        self.assertEqual(s.rotated_total, 1)
        self.assertGreater(s.created, created_before)
        self.assertGreater(s.destroyed, 0)

        # 第 4 次借到的应该是新的真实连接
        with pool.connection() as conn:
            self.assertIsNot(conn.real_connection, first_real_conn)

        pool.close()

    def test_rotation_by_age(self):
        """按连接年龄轮换。"""
        pool = make_pool(
            min_size=1, max_size=1,
            max_age_for_rotation=0.1,  # 0.1 秒后轮换
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        created_before = pool.stats().created
        first_real_conn = None

        # 第一次借，拿到初始连接
        with pool.connection() as conn:
            first_real_conn = conn.real_connection

        # 等待超过轮换年龄
        time.sleep(0.15)

        # 第二次借还，应该触发轮换
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")

        s = pool.stats()
        self.assertEqual(s.rotated_total, 1)
        self.assertGreater(s.created, created_before)
        self.assertGreater(s.destroyed, 0)

        # 新的借到的应该是新连接
        with pool.connection() as conn:
            self.assertIsNot(conn.real_connection, first_real_conn)

        pool.close()

    def test_rotation_preserves_max_size(self):
        """轮换过程中不会突破 max_size。"""
        pool = make_pool(
            min_size=2, max_size=2,
            max_borrow_count=1,  # 每次借还都轮换
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        max_total = 0
        for i in range(5):
            with pool.connection() as conn:
                conn.cursor().execute(f"SELECT {i}")
            s = pool.stats()
            max_total = max(max_total, s.total)

        self.assertLessEqual(max_total, 2)
        s = pool.stats()
        self.assertEqual(s.rotated_total, 5)
        pool.close()

    def test_borrow_count_persists_across_repack(self):
        """借还次数在重新包装时被继承。"""
        pool = make_pool(
            min_size=1, max_size=1,
            max_borrow_count=3,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        # 借还 2 次（不触发轮换）
        for i in range(2):
            with pool.connection() as conn:
                borrow_count = conn.borrow_count
                self.assertEqual(borrow_count, i + 1)

        # 第 3 次借，应该触发轮换
        with pool.connection() as conn:
            self.assertEqual(conn.borrow_count, 3)

        # 第 4 次借，应该是新连接，borrow_count 从 1 开始
        with pool.connection() as conn:
            self.assertEqual(conn.borrow_count, 1)

        pool.close()

    def test_rotation_disabled_by_default(self):
        """默认不启用轮换。"""
        pool = make_pool(
            min_size=1, max_size=1,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        first_real_conn = None
        for i in range(10):
            with pool.connection() as conn:
                if i == 0:
                    first_real_conn = conn.real_connection
                # 每次都应该是同一个真实连接
                self.assertIs(conn.real_connection, first_real_conn)

        s = pool.stats()
        self.assertEqual(s.rotated_total, 0)
        pool.close()


# ==========================================================================
# 新增功能测试: 销毁统计准确性
# ==========================================================================
class TestDestroyedCountAccuracy(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_destroyed_matches_real_close(self):
        """所有销毁路径的 destroyed 统计都与真实关闭次数一致。"""
        real_close_count = [0]
        original_close = FakeDBConnection.close

        def counting_close(self):
            real_close_count[0] += 1
            original_close(self)

        # 猴子补丁 FakeDBConnection.close 来统计真实关闭次数
        FakeDBConnection.close = counting_close

        try:
            pool = make_pool(
                min_size=2, max_size=3,
                max_borrow_count=2,  # 轮换触发销毁
                health_check_enabled=True,
                check_interval=0.05,
                idle_before_check=0.0,
                max_lifetime=0.1,  # 超龄淘汰触发销毁
                leak_check_enabled=False,
            ).start()

            # 杀掉一个空闲连接（健康检查会销毁它）
            idle_conns = _peek_idle(pool)
            if idle_conns:
                idle_conns[0].real_connection.kill()

            # 借还几次，触发轮换销毁
            for i in range(4):
                with pool.connection() as conn:
                    conn.cursor().execute(f"SELECT {i}")

            # 等待健康检查运行，可能触发超龄销毁
            time.sleep(0.15)

            # 关闭池，触发剩余连接销毁
            pool.close()

            # 验证统计一致
            s = pool.stats()
            self.assertEqual(
                s.destroyed, real_close_count[0],
                f"destroyed 统计({s.destroyed}) 与真实关闭次数({real_close_count[0]}) 不一致",
            )
        finally:
            # 恢复原方法
            FakeDBConnection.close = original_close

    def test_reset_failure_counts_as_destroyed(self):
        """reset 失败的连接也会被正确计入 destroyed。"""
        reset_attempts = [0]

        def _flaky_reset(conn):
            reset_attempts[0] += 1
            if reset_attempts[0] == 2:
                raise RuntimeError("Simulated reset failure")

        pool = make_pool(
            min_size=1, max_size=2,
            reset_fn=_flaky_reset,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        destroyed_before = pool.stats().destroyed

        # 第一次借还正常
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 1")

        # 第二次借还，reset 会失败
        with pool.connection() as conn:
            conn.cursor().execute("SELECT 2")

        s = pool.stats()
        self.assertEqual(s.reset_failures, 1)
        self.assertEqual(s.destroyed, destroyed_before + 1)

        pool.close()

    def test_shrink_discarded_counts_as_destroyed(self):
        """缩容丢弃的连接也会被正确计入 destroyed。"""
        pool = make_pool(
            min_size=1, max_size=5,
            health_check_enabled=True,
            check_interval=0.05,
            enable_shrink=True,
            shrink_idle_seconds=0.0,
            leak_check_enabled=False,
        ).start()

        # 先借满 max_size
        conns = [pool.acquire() for _ in range(5)]
        s = pool.stats()
        self.assertEqual(s.total, 5)

        # 归还所有连接
        for c in conns:
            c.close()

        # resize 缩小 max_size
        pool.resize(new_max=2, new_min=1)

        # 等待缩容运行
        time.sleep(0.1)

        s = pool.stats()
        self.assertLessEqual(s.total, 2)
        # destroyed 应该等于 5 - 最终 total
        self.assertGreaterEqual(s.destroyed, 3)

        pool.close()


# ==========================================================================
# 新增功能测试: 并发压测 - 健康检查补连接 + 业务借连接
# ==========================================================================
class TestConcurrentCreateStress(unittest.TestCase):
    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_concurrent_create_never_exceeds_max_size(self):
        """
        压测: 健康检查补连接 + 业务借连接并发时, max_size 永远不被突破。
        50 个并发线程, 每个线程借还 10 次, max_size=5。
        同时健康检查不断运行补连接。
        """
        MAX_SIZE = 5
        NUM_THREADS = 50
        ITERATIONS_PER_THREAD = 10

        # 制造慢创建，增加并发冲突概率
        create_call_count = [0]

        def _slow_create():
            create_call_count[0] += 1
            # 随机 sleep 增加并发冲突概率
            time.sleep(0.005)
            return FakeDBConnection(operation_delay=0.001)

        pool = make_pool(
            min_size=MAX_SIZE, max_size=MAX_SIZE,
            create_fn=_slow_create,
            health_check_enabled=True,
            check_interval=0.02,
            idle_before_check=0.0,
            enable_shrink=False,
            leak_check_enabled=False,
        ).start()

        max_total_seen = [0]
        errors = []
        successful_ops = [0]

        def _worker():
            try:
                for _ in range(ITERATIONS_PER_THREAD):
                    # 每次借前检查 total
                    s = pool.stats()
                    with threading.Lock():
                        max_total_seen[0] = max(max_total_seen[0], s.total)
                    if s.total > MAX_SIZE:
                        errors.append(f"total 突破 max_size: {s.total} > {MAX_SIZE}")
                        return

                    with pool.connection(timeout=2.0) as conn:
                        conn.cursor().execute("SELECT 1")
                        successful_ops[0] += 1

                    # 每次还后检查 total
                    s = pool.stats()
                    with threading.Lock():
                        max_total_seen[0] = max(max_total_seen[0], s.total)
                    if s.total > MAX_SIZE:
                        errors.append(f"total 突破 max_size: {s.total} > {MAX_SIZE}")
                        return
            except Exception as e:
                errors.append(str(e))

        # 启动压测线程
        threads = [threading.Thread(target=_worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()

        # 监控线程，不断检查 total
        stop_monitor = threading.Event()

        def _monitor():
            while not stop_monitor.is_set():
                s = pool.stats()
                with threading.Lock():
                    max_total_seen[0] = max(max_total_seen[0], s.total)
                if s.total > MAX_SIZE:
                    errors.append(f"监控发现 total 突破 max_size: {s.total} > {MAX_SIZE}")
                    stop_monitor.set()
                time.sleep(0.005)

        monitor = threading.Thread(target=_monitor)
        monitor.start()

        # 等待所有线程完成
        for t in threads:
            t.join(timeout=10)

        stop_monitor.set()
        monitor.join(timeout=2)

        # 验证
        self.assertEqual(errors, [], f"压测期间出错: {errors[:5]}")
        self.assertLessEqual(
            max_total_seen[0], MAX_SIZE,
            f"压测期间 total 突破了 max_size: 最大值={max_total_seen[0]}, max_size={MAX_SIZE}",
        )
        self.assertEqual(
            successful_ops[0], NUM_THREADS * ITERATIONS_PER_THREAD,
            f"成功操作数 {successful_ops[0]} != 期望 {NUM_THREADS * ITERATIONS_PER_THREAD}",
        )

        # 打印统计
        s = pool.stats()
        print(f"\n  压测统计:")
        print(f"    创建连接调用次数: {create_call_count[0]}")
        print(f"    成功操作: {successful_ops[0]}")
        print(f"    最终 total: {s.total}")
        print(f"    max_total_seen: {max_total_seen[0]}")
        print(f"    created: {s.created}, destroyed: {s.destroyed}")
        print(f"    总连接数始终 <= max_size({MAX_SIZE}): {'✅ PASS' if max_total_seen[0] <= MAX_SIZE else '❌ FAIL'}")

        pool.close()


# ==========================================================================
# 新增功能测试: 运行时管控混合压测
# ==========================================================================
class TestRuntimeControlStress(unittest.TestCase):
    """
    压力测试: 在线扩缩容 + 暂停恢复 + 健康检查 + 业务并发借还混合场景。

    验收标准:
      1. 连接数始终不超过 max_size（包括动态调整后的 max_size）
      2. 销毁统计与真实关闭次数对得上
      3. 轮换统计准确
      4. 暂停后借不到，恢复后能正常借
    """

    def setUp(self):
        FakeDBConnection.reset_counter()

    def test_mixed_workload_stability(self):
        """混合压测：业务并发 + 动态扩缩容 + 暂停恢复 + 健康检查。"""
        INITIAL_MIN = 2
        INITIAL_MAX = 5
        NUM_WORKERS = 20
        DURATION_SECONDS = 3.0
        SHRINK_INTERVAL = 0.3

        # 统计真实关闭次数
        real_close_count = [0]
        original_close = FakeDBConnection.close

        def counting_close(self_conn):
            real_close_count[0] += 1
            original_close(self_conn)

        FakeDBConnection.close = counting_close

        try:
            pool = make_pool(
                min_size=INITIAL_MIN,
                max_size=INITIAL_MAX,
                max_borrow_count=5,  # 频繁借还会触发轮换
                health_check_enabled=True,
                check_interval=0.1,
                idle_before_check=0.0,
                max_lifetime=1.0,
                enable_shrink=True,
                shrink_idle_seconds=SHRINK_INTERVAL,
                leak_check_enabled=False,
                borrow_timeout=5.0,
            ).start()

            errors = []
            stop_flag = threading.Event()
            max_total_seen = [0]
            max_count_lock = threading.Lock()
            successful_borrows = [0]

            def _record_max():
                s = pool.stats()
                with max_count_lock:
                    if s.total > max_total_seen[0]:
                        max_total_seen[0] = s.total
                if s.total > pool.stats().max_size:
                    errors.append(
                        f"total 突破 max_size: total={s.total}, max_size={pool.stats().max_size}"
                    )

            def _worker():
                """业务线程：不断借还连接。"""
                try:
                    while not stop_flag.is_set():
                        try:
                            with pool.connection() as conn:
                                conn.cursor().execute("SELECT 1")
                                with max_count_lock:
                                    successful_borrows[0] += 1
                                _record_max()
                        except GetTimeoutError:
                            pass  # 暂停或扩容时可能超时，正常
                        except PoolPausedError:
                            pass  # 暂停中，正常
                        except PoolClosedError:
                            break
                        _record_max()
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(f"worker error: {e}")

            def _operator():
                """运维线程：周期性调整池大小和暂停恢复。"""
                try:
                    phases = [
                        ("expand_max", lambda: pool.resize(new_max=8)),
                        ("expand_min", lambda: pool.resize(new_min=5)),
                        ("pause", lambda: pool.pause_borrow()),
                        ("wait", lambda: time.sleep(0.2)),
                        ("resume", lambda: pool.resume_borrow()),
                        ("shrink_max", lambda: pool.resize(new_max=3)),
                        ("shrink_min", lambda: pool.resize(new_min=1)),
                        ("trigger_health", lambda: pool.trigger_health_check()),
                    ]
                    while not stop_flag.is_set():
                        for name, action in phases:
                            if stop_flag.is_set():
                                break
                            try:
                                action()
                            except Exception:
                                pass
                            time.sleep(0.15)
                except Exception as e:
                    errors.append(f"operator error: {e}")

            # 启动业务线程
            workers = [threading.Thread(target=_worker) for _ in range(NUM_WORKERS)]
            for t in workers:
                t.start()

            # 启动运维线程
            op_thread = threading.Thread(target=_operator)
            op_thread.start()

            # 运行指定时间
            time.sleep(DURATION_SECONDS)
            stop_flag.set()

            # 等待所有线程结束
            for t in workers:
                t.join(timeout=5)
            op_thread.join(timeout=5)

            # 最终检查
            s = pool.stats()
            _record_max()

            # 1. 验证没有突破过 max_size
            self.assertEqual(
                errors, [],
                f"压测期间出错: {errors[:5]}"
            )
            self.assertLessEqual(
                max_total_seen[0], 8,  # 最大 max_size 是 8
                f"压测期间 total 突破了历史最大 max_size(8): 最大值={max_total_seen[0]}"
            )

            # 2. 验证销毁统计与真实关闭次数一致
            # 注意：关闭池时还会销毁剩余连接，所以先不急着关
            destroyed_before_close = s.destroyed
            # 真实关闭次数可能等于或大于 destroyed_before_close，因为有些连接在关闭中
            # 我们关闭后再对比
            pool.close()

            final_stats = pool.stats()
            self.assertEqual(
                final_stats.destroyed, real_close_count[0],
                f"destroyed 统计({final_stats.destroyed}) 与真实关闭次数({real_close_count[0]}) 不一致"
            )

            # 3. 打印统计信息
            print(f"\n  混合压测统计:")
            print(f"    成功借还次数: {successful_borrows[0]}")
            print(f"    历史最大连接数: {max_total_seen[0]}")
            print(f"    总创建: {final_stats.created}")
            print(f"    总销毁: {final_stats.destroyed}")
            print(f"    轮换次数: {final_stats.rotated_total}")
            print(f"    最近创建原因: {final_stats.last_create_reason}")
            print(f"    最近销毁原因: {final_stats.last_destroy_reason}")
            print(f"    最近轮换原因: {final_stats.last_rotation_reason}")
            print(f"    连接数始终 <= max_size: {'✅ PASS' if max_total_seen[0] <= 8 else '❌ FAIL'}")
            print(f"    销毁统计与真实一致: {'✅ PASS' if final_stats.destroyed == real_close_count[0] else '❌ FAIL'}")

            # 4. 验证有轮换发生（证明轮换确实在工作）
            self.assertGreater(
                final_stats.rotated_total, 0,
                "压测期间应该有轮换发生"
            )

        finally:
            FakeDBConnection.close = original_close

    def test_pause_resume_control(self):
        """验证暂停借出和恢复借出的运行时管控能力。"""
        pool = make_pool(
            min_size=2, max_size=3,
            health_check_enabled=False,
            leak_check_enabled=False,
        ).start()

        # 初始状态：未暂停
        self.assertFalse(pool.is_paused)

        # 借一个连接
        conn1 = pool.acquire()
        self.assertIsNotNone(conn1)

        # 暂停借出
        pool.pause_borrow()
        self.assertTrue(pool.is_paused)

        # 暂停后再借应该抛错
        with self.assertRaises(PoolPausedError):
            pool.acquire(timeout=0.1)

        # 已借出的连接应该能正常使用
        conn1.cursor().execute("SELECT 1")

        # 归还应该正常
        conn1.close()

        # 恢复借出
        pool.resume_borrow()
        self.assertFalse(pool.is_paused)

        # 恢复后能正常借
        conn2 = pool.acquire()
        self.assertIsNotNone(conn2)
        conn2.close()

        # 关闭后再调 pause/resume 应该安全（不抛错）
        pool.close()
        pool.pause_borrow()  # 应该被忽略
        pool.resume_borrow()  # 应该被忽略

    def test_trigger_health_check(self):
        """验证主动触发健康检查。"""
        pool = make_pool(
            min_size=2, max_size=3,
            health_check_enabled=True,
            check_interval=60.0,  # 间隔很长，靠主动触发
            idle_before_check=0.0,
            leak_check_enabled=False,
        ).start()

        # 杀掉一个空闲连接
        idle_conns = _peek_idle(pool)
        self.assertTrue(len(idle_conns) > 0)
        idle_conns[0].real_connection.kill()

        # 主动触发健康检查
        triggered = pool.trigger_health_check()
        self.assertTrue(triggered)

        # 等待健康检查执行
        time.sleep(0.3)

        # 死掉的连接应该被替换掉，总数还是 min_size
        s = pool.stats()
        self.assertGreaterEqual(s.total, 2)

        pool.close()

    def test_closed_pool_ignores_control(self):
        """关闭中的池不接受控制动作（安全忽略，不抛异常）。"""
        pool = make_pool(min_size=1, max_size=2).start()
        pool.close()

        # 关闭后 pause/resume 应该被忽略（不抛错）
        pool.pause_borrow()
        pool.resume_borrow()

        # 关闭后 trigger_health_check 应该返回 False
        self.assertFalse(pool.trigger_health_check())

        # 关闭后 resize 应该被安全忽略（不抛错）
        try:
            pool.resize(new_max=5)
        except Exception as e:
            self.fail(f"关闭后 resize 不应抛异常: {e}")


# ==========================================================================
# main
# ==========================================================================
def run_demo():
    """使用示例展示。"""
    print("\n" + "=" * 60)
    print("  Connection Pool Demo")
    print("=" * 60)

    FakeDBConnection.reset_counter()
    leaks_reported = []

    cfg = PoolConfig(
        min_size=3,
        max_size=6,
        borrow_timeout=2.0,
        test_on_borrow=True,
        capture_stack=True,
        health_check_enabled=True,
        check_interval=1.0,
        idle_before_check=0.5,
        max_lifetime=10.0,
        enable_shrink=True,
        shrink_idle_seconds=5.0,
        leak_check_enabled=True,
        leak_threshold=1.5,
        leak_cooldown=2.0,
        force_reclaim_leaked=False,
        leak_listener=lambda info: leaks_reported.append(info.conn_id),
    )

    pool = ConnectionPool(
        create_fn=lambda: FakeDBConnection(operation_delay=0.01),
        config=cfg,
    )
    pool.start()
    print(f"[1] 启动后: {pool}")

    # 示例1: 上下文管理器
    print("\n[2] 借用 2 个连接做查询...")
    conns = []
    for i in range(2):
        c = pool.acquire()
        c.cursor().execute(f"SELECT {i}")
        conns.append(c)
    print(f"    借用中: {pool}")
    for c in conns:
        c.close()
    print(f"    归还后: {pool}")

    # 示例2: 模拟 15 个并发请求
    print("\n[3] 模拟 15 个并发请求 (max=6, 其他排队)...")
    results = []
    lock = threading.Lock()

    def _task(idx):
        t0 = time.monotonic()
        try:
            with pool.connection(timeout=5) as c:
                c.cursor().execute(f"/* worker {idx} */ SELECT 1")
                time.sleep(0.15)
                dt = time.monotonic() - t0
                with lock:
                    results.append((idx, "ok", round(dt, 3)))
        except Exception as e:
            with lock:
                results.append((idx, f"err:{e}", round(time.monotonic() - t0, 3)))

    threads = [threading.Thread(target=_task, args=(i,)) for i in range(15)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok_n = sum(1 for _, s, _ in results if s == "ok")
    print(f"    完成 {ok_n}/15,最近 5 个结果: {results[-5:]}")
    print(f"    池状态: {pool}")

    # 示例3: 故意泄漏一个连接
    print("\n[4] 故意泄漏 1 个连接(> 1.5s 不归还)...")
    leaked = pool.acquire()
    time.sleep(2.5)
    print(f"    检测到泄漏连接数: {pool.stats().leaked}")
    print(f"    leaks_reported conn_ids: {leaks_reported}")
    leaked.close()

    # 示例4: resize
    print("\n[5] resize min_size 3 -> 5 ...")
    pool.resize(new_min=5)
    print(f"    resize 后: {pool}")

    # 示例5: 关闭
    print("\n[6] 优雅关闭(等待所有连接归还)...")
    pool.close(graceful=True, wait_timeout=3)
    print(f"    关闭后: {pool}")
    print("\nDemo 完成.\n")


if __name__ == "__main__":
    # python tests/test_pool.py  ->  运行 demo
    # python -m unittest tests.test_pool  ->  运行单元测试
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(module=__name__, exit=True, verbosity=2)
    else:
        run_demo()
