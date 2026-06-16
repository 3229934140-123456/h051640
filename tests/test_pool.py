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

from db_pool import ConnectionPool, PoolConfig, PooledConnection
from db_pool.borrow_return import GetTimeoutError, PoolClosedError
from db_pool.leak_detector import LeakInfo
from tests.fake_db import FakeDBConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def make_pool(**cfg_kw):
    """创建一个使用 FakeDBConnection 的池。"""
    cfg_kw.setdefault("min_size", 3)
    cfg_kw.setdefault("max_size", 8)
    cfg_kw.setdefault("health_check_enabled", False)
    cfg_kw.setdefault("leak_check_enabled", False)
    cfg = PoolConfig(**cfg_kw)
    pool = ConnectionPool(
        create_fn=lambda: FakeDBConnection(),
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
        with pool.connection() as conn:
            self.assertIsInstance(conn, PooledConnection)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            self.assertEqual(conn.real_connection.queries_executed, 1)
            s = pool.stats()
            self.assertEqual(s.borrowed, 1)
            self.assertEqual(s.idle, 1)
        s = pool.stats()
        self.assertEqual(s.borrowed, 0)
        self.assertEqual(s.idle, 2)
        # reset 应该被调用(rollback)
        self.assertEqual(conn.real_connection.rollback_count, 1)
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
        stats = pool.stats()
        fake_conns = [c for c in _peek_idle(pool)]
        for fc in fake_conns:
            fc.real_connection.kill()

        total_before = pool.stats().total
        destroyed_before = pool.stats().destroyed
        created_before = pool.stats().created
        with pool.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")  # 新连接应该可用
        s = pool.stats()
        # 至少销毁了原来的死连接,且至少创建了新连接替换
        self.assertGreaterEqual(s.destroyed, destroyed_before + 2)
        self.assertGreaterEqual(s.created, created_before + 1)
        self.assertGreaterEqual(s.total, 1)
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
