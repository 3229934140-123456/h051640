"""
借还逻辑模块
============

同步策略 (关键设计):
- 所有对 idle/borrowed/waiters 的访问都在 Condition 保护下。
- Condition 底层使用 PoolManager 的 RLock,可以重入,但为了避免和
  `with self._cond:` 冲突,我们只用 `with self._cond:` 进入临界区,
  绝不手动调用 release()/acquire()。
- 创建连接(可能走网络)在锁外执行,但"决定能否创建"的名额判断在锁内:
  通过 `_pending_creates` 计数器占坑,保证并发创建总数也不超过 max_size。
- ping / reset / destroy 也在锁外执行,避免 I/O 阻塞其他借还线程。

借出流程:
    1. [锁内] 弹 idle → 有就拿
    2. [锁内] 没 idle 但 `total + _pending_creates < max_size` → 占坑,出锁创建
    3. [锁内] 否则 → Condition.wait(deadline),被唤醒重走 1-2,超时抛错
    4. [锁外] ping 校验 → 失败则销毁重走 1-3
    5. [锁内] 登记 borrowed,返回连接

归还流程:
    1. [锁内] 从 borrowed 注销
    2. [锁外] factory.reset
    3. [锁内] 入池或销毁 → notify 一个等待者
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

from .connection import PooledConnection
from .connection_factory import ConnectionFactory
from .pool_manager import PoolManager


logger = logging.getLogger("db_pool.borrow")

_TAKE_IDLE = "idle"
_TAKE_CREATE = "create"


class GetTimeoutError(TimeoutError):
    """在 borrow_timeout 内未能获取到连接。"""


class PoolClosedError(RuntimeError):
    """池已关闭,拒绝借出或归还操作。"""


class BorrowReturn:
    def __init__(
        self,
        manager: PoolManager,
        factory: ConnectionFactory,
        *,
        borrow_timeout: float = 8.0,
        test_on_borrow: bool = True,
        capture_stack_on_borrow: bool = False,
    ) -> None:
        self._manager = manager
        self._factory = factory
        self._borrow_timeout = borrow_timeout
        self._test_on_borrow = test_on_borrow
        self._capture_stack = capture_stack_on_borrow

        self._lock = manager.acquire_lock()
        self._cond = threading.Condition(self._lock)
        self._waiters: Deque[threading.Event] = deque()
        self._pending_creates: int = 0

    # ---------------------------------------------------------- 指标

    @property
    def waiting_count(self) -> int:
        with self._cond:
            return len(self._waiters)

    # ---------------------------------------------------------- 借出

    def borrow(self, timeout: Optional[float] = None) -> PooledConnection:
        if self._manager.is_shutdown:
            raise PoolClosedError("Connection pool is closed")

        effective_timeout = self._borrow_timeout if timeout is None else timeout
        deadline = time.monotonic() + effective_timeout

        while True:
            # --- Phase A: 锁内决定获取方式 (idle / 创建名额 / 排队) ---
            kind, candidate = self._acquire_slot_locked(deadline, effective_timeout)

            if kind == _TAKE_CREATE:
                # candidate is None; 我们占了创建坑,出锁去创建
                new_conn = self._create_out_of_lock()
                if new_conn is None:
                    # 创建失败, 释放占坑,重新循环尝试
                    with self._cond:
                        self._pending_creates -= 1
                        self._wake_one_locked()  # 让别人试试
                    continue
                candidate = new_conn
                # 注意: 创建完成后,占坑的 _pending_creates 还要在 register_borrowed 时减去
                # 我们在 register_borrowed_locked 里处理

            # --- Phase B: 锁外健康校验 (candidate 一定不是 None) ---
            if not self._validate_out_of_lock(candidate):
                self._factory.destroy(candidate)
                with self._cond:
                    if kind == _TAKE_CREATE:
                        self._pending_creates -= 1
                    self._manager._stats.destroyed += 1
                    self._wake_one_locked()
                continue

            # --- Phase C: 锁内完成登记 ---
            final = self._register_borrowed_locked(candidate, kind)
            if final is not None:
                return final
            # final 为 None → 中途 shutdown, 销毁 candidate 继续
            self._factory.destroy(candidate)

    # ------------------------------------------------------------------
    # Phase A 辅助: 锁内获取一个"坑"
    # ------------------------------------------------------------------
    def _acquire_slot_locked(
        self, deadline: float, original_timeout: float,
    ) -> Tuple[str, Optional[PooledConnection]]:
        """
        返回:
          - (_TAKE_IDLE, conn)       : 从 idle 拿到了候选
          - (_TAKE_CREATE, None)     : 抢到了一个创建名额,需要出锁创建
        否则等待到超时抛错。
        """
        with self._cond:
            while True:
                if self._manager.is_shutdown:
                    raise PoolClosedError("Connection pool is closed")

                # a) 先弹 idle
                c = self._manager.pop_idle()
                if c is not None:
                    return (_TAKE_IDLE, c)

                # b) 是否能创建 (total + 正在创建的 < max)?
                total = (
                    self._manager.idle_count
                    + self._manager.borrowed_count
                    + self._pending_creates
                )
                if total < self._manager._max_size:
                    self._pending_creates += 1
                    return (_TAKE_CREATE, None)

                # c) 排队
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._manager.inc_timeout()
                    s = self._manager.snapshot_stats(waiting=len(self._waiters))
                    raise GetTimeoutError(
                        f"Timed out waiting for connection after "
                        f"{original_timeout:.2f}s, "
                        f"total={s.total} idle={s.idle} borrowed={s.borrowed} "
                        f"waiting={s.waiting}"
                    )

                ev = threading.Event()
                self._waiters.append(ev)
                try:
                    # 等待外部 notify_all / 超时。 Condition.wait 会释放/重拿锁
                    self._cond.wait(timeout=remaining)
                finally:
                    try:
                        self._waiters.remove(ev)
                    except ValueError:
                        pass

    # ------------------------------------------------------------------
    # Phase B 辅助: 创建 & 校验 (锁外)
    # ------------------------------------------------------------------
    def _create_out_of_lock(self) -> Optional[PooledConnection]:
        try:
            c = self._factory.create(self._manager._pool_ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Create connection failed: %s", exc)
            return None
        with self._lock:
            self._manager._stats.created += 1
        return c

    def _validate_out_of_lock(self, c: PooledConnection) -> bool:
        if not self._test_on_borrow:
            return True
        return self._factory.ping(c)

    # ------------------------------------------------------------------
    # Phase C 辅助: 最终登记 (锁内)
    # ------------------------------------------------------------------
    def _register_borrowed_locked(
        self, c: PooledConnection, kind: str
    ) -> Optional[PooledConnection]:
        """成功登记返回连接,中途 shutdown 返回 None。"""
        with self._cond:
            if kind == _TAKE_CREATE:
                self._pending_creates -= 1
            if self._manager.is_shutdown:
                return None
            c.mark_borrowed(capture_stack=self._capture_stack)
            c._is_closed = False
            self._manager.register_borrowed(c)
            return c

    # ---------------------------------------------------------- 归还

    def release(self, conn: PooledConnection) -> None:
        if conn is None:
            return

        # 1) 锁内从 borrowed 注销
        with self._cond:
            existed = self._manager.unregister_borrowed(conn)
            if existed is None:
                logger.warning(
                    "Returned connection %s not in borrowed set, ignoring",
                    conn.conn_id,
                )
                return
            conn.mark_returned()

        # 2) 锁外 reset (可能包含 DB 回滚等 I/O)
        reset_ok = self._factory.reset(conn)

        # 3) 锁内决定去向 + 唤醒
        with self._cond:
            if not reset_ok:
                self._factory.destroy(conn)
                self._manager._stats.destroyed += 1
                self._wake_one_locked()
                return

            if self._manager.is_shutdown:
                self._factory.destroy(conn)
                self._manager._stats.destroyed += 1
                self._wake_one_locked()
                return

            if self._manager.total_count >= self._manager._max_size:
                # 动态缩容后的超额连接
                self._factory.destroy(conn)
                self._manager._stats.destroyed += 1
                self._wake_one_locked()
                return

            conn._is_closed = False
            self._manager.add_idle(conn)
            self._wake_one_locked()

    # ---------------------------------------------------------- 唤醒

    def _wake_one_locked(self) -> None:
        """
        从 FIFO 队首唤醒一位 Event 等待者,同时 notify_all 唤醒通过
        Condition.wait 等待的线程(例如 wait_until_all_returned)。
        必须已持锁。
        """
        while self._waiters:
            ev = self._waiters.popleft()
            if not ev.is_set():
                ev.set()
                break
        # 一定要调用 notify_all,因为除了 Event 队列里的等待者,
        # wait_until_all_returned 等场景直接使用 Condition.wait(),
        # 它们不在 _waiters 里,必须被 notify 才能醒来。
        self._cond.notify_all()

    # ---------------------------------------------------------- 优雅关闭

    def wait_until_all_returned(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._manager.borrowed_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
        return True


