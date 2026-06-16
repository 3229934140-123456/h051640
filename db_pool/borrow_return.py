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

归还流程 (关键: 连接剥离设计):
    1. [锁内] 从 borrowed 注销
    2. [锁外] strip_real_connection 剥离底层真实连接
    3. [锁外] factory.reset_real 重置真实连接
    4. [锁外] 成功 → factory.wrap_real 重新包装为新的 PooledConnection
                失败 → factory.destroy_real 销毁真实连接
    5. [锁内] 新包装好的连接入池 或 直接销毁 → notify_all

连接剥离设计保证:
    - 用户持有的旧 PooledConnection 引用已标记 returned=True,任何访问都会
      抛 ConnectionReturnedError,防止连接归还后继续被误用。
    - 底层真实连接被新的 PooledConnection 包装后可以安全复用。
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
            wait_start = time.monotonic()

            # --- Phase A: 锁内决定获取方式 (idle / 创建名额 / 排队) ---
            kind, candidate = self._acquire_slot_locked(deadline, effective_timeout)

            # 记录等待耗时 (从进入 borrow 到拿到候选)
            wait_time = time.monotonic() - wait_start
            if wait_time > 0:
                self._manager.record_wait_time(wait_time)

            if kind == _TAKE_CREATE:
                # candidate is None; 我们占了创建坑,出锁去创建
                new_conn = self._create_out_of_lock()
                if new_conn is None:
                    # 创建失败, 释放占坑,重新循环尝试
                    with self._cond:
                        self._pending_creates -= 1
                        self._wake_one_locked()
                    continue
                candidate = new_conn

            # --- Phase B: 锁外健康校验 (candidate 一定不是 None) ---
            if not self._validate_out_of_lock(candidate):
                # 使用 manager.destroy_connection 确保 destroyed 统计被正确更新
                self._manager.destroy_connection(candidate)
                with self._cond:
                    if kind == _TAKE_CREATE:
                        self._pending_creates -= 1
                    self._wake_one_locked()
                continue

            # --- Phase C: 锁内完成登记 ---
            final = self._register_borrowed_locked(candidate, kind)
            if final is not None:
                return final
            # final 为 None → 中途 shutdown, 销毁 candidate 继续
            # 使用 manager.destroy_connection 确保 destroyed 统计被正确更新
            self._manager.destroy_connection(candidate)

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

                # b) 是否能创建 (total + 正在创建的 + 健康检查中的 < max)?
                total = (
                    self._manager.idle_count
                    + self._manager.borrowed_count
                    + self._pending_creates
                    + self._manager.in_health_check_count
                )
                if total < self._manager._max_size:
                    self._pending_creates += 1
                    return (_TAKE_CREATE, None)

                # c) 排队
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._manager.inc_timeout()
                    s = self._manager.snapshot_stats(
                        waiting=len(self._waiters),
                        pending_creates=self._pending_creates,
                    )
                    raise GetTimeoutError(
                        f"Timed out waiting for connection after "
                        f"{original_timeout:.2f}s, "
                        f"total={s.total} idle={s.idle} borrowed={s.borrowed} "
                        f"in_health_check={s.in_health_check} "
                        f"pending_creates={s.pending_creates} "
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
            # 注意: create_failures 由 ConnectionFactory 通过 stats_callback 报告
            return None
        # 增加 created 统计
        with self._cond:
            self._manager._stats.created += 1
        return c

    def _validate_out_of_lock(self, c: PooledConnection) -> bool:
        if not self._test_on_borrow:
            return True
        ok = self._factory.ping(c)
        # 注意: ping_failures 由 ConnectionFactory 通过 stats_callback 报告
        return ok

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
            self._manager.register_borrowed(c)
            return c

    # ---------------------------------------------------------- 归还 (连接剥离流程)

    def release(self, conn: PooledConnection) -> None:
        """
        归还连接,采用"连接剥离"设计:
        1) 锁内从 borrowed 注销
        2) 锁外从旧 PooledConnection 剥离真实连接
        3) 锁外 reset 真实连接
        4) 锁外用真实连接创建新的 PooledConnection 包装
        5) 锁内判断入池或销毁
        """
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

        # 2) 锁外剥离真实连接 (旧包装对象永久失效)
        old_conn_id = conn.conn_id
        real_conn = conn.strip_real_connection()
        if real_conn is None:
            logger.warning(
                "Returned connection id=%s has no real connection, ignoring",
                old_conn_id,
            )
            with self._cond:
                self._wake_one_locked()
            return

        # 3) 锁外 reset 真实连接
        reset_ok = self._factory.reset_real(real_conn, old_conn_id)
        if not reset_ok:
            # 注意: reset_failures 由 ConnectionFactory 通过 stats_callback 报告
            self._factory.destroy_real(real_conn)
            with self._cond:
                self._wake_one_locked()
            return

        # 4) 锁外用真实连接创建新的 PooledConnection 包装
        #    注意: 这里不持锁,但创建新 PooledConnection 只是内存操作很快
        new_pooled = self._factory.wrap_real(real_conn, self._manager._pool_ref)

        # 5) 锁内决定去向 + 唤醒
        with self._cond:
            if self._manager.is_shutdown:
                self._factory.destroy_real(real_conn)
                self._wake_one_locked()
                return

            total = (
                self._manager.idle_count
                + self._manager.borrowed_count
                + self._pending_creates
                + self._manager.in_health_check_count
            )
            if total >= self._manager._max_size:
                # 动态缩容后的超额连接
                self._factory.destroy_real(real_conn)
                self._wake_one_locked()
                return

            self._manager.add_idle(new_pooled)
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

    def wait_until_all_returned(self, timeout: float = 30.0) -> Tuple[bool, list[int]]:
        """
        在已经 begin_shutdown 之后调用:
        - 新来的 borrow 直接抛 PoolClosedError;
        - 归还流程正常走,每归还一个通知本线程,直到 borrowed==0 或超时。

        返回: (是否全部归还, 超时时仍未归还的 conn_id 列表)
        """
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._manager.borrowed_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 收集未归还的 conn_id 用于诊断
                    outstanding = list(self._manager._borrowed.keys())
                    return False, outstanding
                self._cond.wait(timeout=remaining)
        return True, []
