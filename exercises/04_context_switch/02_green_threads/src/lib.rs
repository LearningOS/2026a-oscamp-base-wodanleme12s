//! # Green Thread Scheduler (riscv64)
//!
//! In this exercise, you build a simple cooperative (green) thread scheduler on top of context switching.
//! This crate is **riscv64 only**; run with the repo's normal flow (`./check.sh` / `oscamp`) or natively on riscv64.
//!
//! ## Key Concepts
//! - Cooperative vs preemptive scheduling
//! - Thread state: `Ready`, `Running`, `Finished`
//! - `yield_now()`: current thread voluntarily gives up the CPU
//! - Scheduler loop: pick next ready thread and switch to it
//!
//! ## Design
//! Each green thread has its own stack and `TaskContext`. Threads call `yield_now()` to yield.
//! The scheduler round-robins among ready threads. User entry is wrapped by `thread_wrapper`, which
//! calls the entry then marks the thread `Finished` and switches back.

#![cfg(target_arch = "riscv64")]

use core::arch::naked_asm;

/// Per-thread stack size. Slightly larger to avoid overflow under QEMU / test harness.
const STACK_SIZE: usize = 1024 * 128;

/// Task context (riscv64); layout must match `01_stack_coroutine::TaskContext` and the asm below.
#[repr(C)]
#[derive(Debug, Default, Clone)]
pub struct TaskContext {
    sp: u64,
    ra: u64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    s5: u64,
    s6: u64,
    s7: u64,
    s8: u64,
    s9: u64,
    s10: u64,
    s11: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadState {
    Ready,
    Running,
    Finished,
}

struct GreenThread {
    ctx: TaskContext,
    state: ThreadState,
    _stack: Option<Vec<u8>>,
    /// User entry; taken once when the thread is first scheduled and passed to `thread_wrapper`.
    entry: Option<extern "C" fn()>,
}

/// 调度器在切入新线程前写入，`thread_wrapper` 读取并只调用一次。
///
/// 为什么不用现成的写法（把 entry 指针压进新线程栈顶，靠 `ret` 逐级弹出）：那种做法要求
/// 我们精确伪造一个"看起来像正常函数调用"的栈帧，编译器一改内联/序言策略就会静默失效。
/// 这里改为**单一交接槽位**：wrapper 本身就是 ra，入口通过全局槽传递，栈里不需要任何伪造帧。
static mut CURRENT_THREAD_ENTRY: Option<extern "C" fn()> = None;

/// Wrapper run as the initial `ra` for each green thread: call the user entry (from `CURRENT_THREAD_ENTRY`), then mark Finished and switch back.
extern "C" fn thread_wrapper() {
    let entry = unsafe { core::ptr::read(&raw const CURRENT_THREAD_ENTRY) };
    if let Some(f) = entry {
        unsafe { CURRENT_THREAD_ENTRY = None };
        f();
    }
    thread_finished();
}

/// Save current callee-saved regs into `old`, load from `new`, then `ret` to `new.ra`.
/// Zero `a0`/`a1` before `ret` so we don't leak pointers into the new context.
///
/// Must be `#[unsafe(naked)]` to prevent the compiler from generating a prologue/epilogue.
#[unsafe(naked)]
unsafe extern "C" fn switch_context(_old: &mut TaskContext, _new: &TaskContext) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd ra, 8(a0)",
        "sd s0, 16(a0)",
        "sd s1, 24(a0)",
        "sd s2, 32(a0)",
        "sd s3, 40(a0)",
        "sd s4, 48(a0)",
        "sd s5, 56(a0)",
        "sd s6, 64(a0)",
        "sd s7, 72(a0)",
        "sd s8, 80(a0)",
        "sd s9, 88(a0)",
        "sd s10, 96(a0)",
        "sd s11, 104(a0)",
        "ld sp, 0(a1)",
        "ld ra, 8(a1)",
        "ld s0, 16(a1)",
        "ld s1, 24(a1)",
        "ld s2, 32(a1)",
        "ld s3, 40(a1)",
        "ld s4, 48(a1)",
        "ld s5, 56(a1)",
        "ld s6, 64(a1)",
        "ld s7, 72(a1)",
        "ld s8, 80(a1)",
        "ld s9, 88(a1)",
        "ld s10, 96(a1)",
        "ld s11, 104(a1)",
        "li a0, 0",
        "li a1, 0",
        "ret",
    );
}

pub struct Scheduler {
    threads: Vec<GreenThread>,
    current: usize,
}

impl Scheduler {
    pub fn new() -> Self {
        let main_thread = GreenThread {
            ctx: TaskContext::default(),
            state: ThreadState::Running,
            _stack: None,
            entry: None,
        };

        Self {
            threads: vec![main_thread],
            current: 0,
        }
    }

    /// Register a new green thread that will run `entry` when first scheduled.
    ///
    /// 1. Allocate a stack of `STACK_SIZE` bytes; compute `stack_top` (high address).
    /// 2. Set up the context: `ra = thread_wrapper` so the first switch jumps to the wrapper;
    ///    `sp` must be 16-byte aligned (e.g. `(stack_top - 16) & !15` to leave headroom).
    /// 3. Push a `GreenThread` with this context, state `Ready`, and `entry` stored for the wrapper to call.
    pub fn spawn(&mut self, entry: extern "C" fn()) {
        let stack = vec![0u8; STACK_SIZE];
        // 栈向下增长：sp 取缓冲区末尾并向 16 字节掩齐。wrapper 是个正常的 extern "C" 函数，
        // 会自己建立序言，所以切入时 sp 只需满足 ABI 的对齐要求，不必预留返回地址槽位。
        let stack_top = stack.as_ptr() as usize + STACK_SIZE;
        let mut ctx = TaskContext::default();
        ctx.sp = (stack_top & !0xF) as u64;
        ctx.ra = thread_wrapper as *const () as u64;

        self.threads.push(GreenThread {
            ctx,
            state: ThreadState::Ready,
            _stack: Some(stack),
            entry: Some(entry),
        });
    }

    /// Run the scheduler until all threads (except the main one) are `Finished`.
    ///
    /// 1. Set the global `SCHEDULER` pointer to `self` so that `yield_now` and `thread_finished` can call back.
    /// 2. Loop: if all threads in `threads[1..]` are `Finished`, break; otherwise call `schedule_next()` (which may switch away and later return).
    /// 3. Clear `SCHEDULER` when done.
    pub fn run(&mut self) {
        // yield_now()/thread_finished() 是无参数自由函数，只能靠这个全局指针回调回来。
        unsafe {
            SCHEDULER = self as *mut Scheduler;
        }

        // 终止条件用"就绪集合是否为空"而不是"是否全部 Finished"：主线程自己不在轮转里，
        // 而它一旦被切出去就可能再也不会回来执行这个判断。改成先数一遍还有没有可跑的子线程，
        // 没有了就直接返回，避免依赖"从子线程切回主线程后继续循环"这一脆弱前提。
        loop {
            let runnable = self
                .threads
                .iter()
                .skip(1)
                .any(|t| t.state != ThreadState::Finished);
            if !runnable {
                break;
            }

            // schedule_next 可能根本不切换（例如只剩一个就绪线程且它就是当前线程）。
            // 若切换成功后控制权回到这里，说明某个子线程让出或结束了，继续轮转即可。
            self.schedule_next();
        }

        unsafe {
            SCHEDULER = core::ptr::null_mut();
        }
    }

    /// Find the next ready thread (starting from `current + 1` round-robin), mark current as `Ready` (if not `Finished`), mark next as `Running`, set `CURRENT_THREAD_ENTRY` if the next thread has an entry, then switch to it.
    fn schedule_next(&mut self) {
        let len = self.threads.len();
        // 环形扫描的范围是 1..=len：包含 offset == len，也就是"重新考虑自己"。
        // 这一点很关键——当唯一一个就绪线程就是当前正在跑的线程时（task_b 结束后只剩
        // task_a），旧写法会找不到目标而原地返回，主循环随即空转；把自身纳入候选后，
        // 该线程只是继续执行，等它下一次 yield 时自然推进到结束判定。
        for offset in 1..=len {
            let next = (self.current + offset) % len;
            if self.threads[next].state == ThreadState::Ready {
                let previous = self.current;
                if self.threads[previous].state != ThreadState::Finished {
                    self.threads[previous].state = ThreadState::Ready;
                }
                self.threads[next].state = ThreadState::Running;
                self.current = next;

                // 首次被调度时把入口交给 wrapper；之后 take() 返回 None，wrapper 不会再调用它。
                unsafe {
                    CURRENT_THREAD_ENTRY = self.threads[next].entry.take();
                }

                // 自切换（next == current）必须跳过 switch_context：此时 old 与 new 是同一块
                // 内存，先把 sp/ra 写进去再从同处读回虽然看似无害，但紧接着的 `ret` 会跳到刚
                // 被覆写的 ra（即 wrapper 本身），造成入口被重复调用。直接返回即可。
                if previous != next {
                    // 借用检查：old 需要 &mut、new 需要 &，二者指向同一 Vec 的不同元素，
                    // 因此拆成两个裸指针再构造引用，语义上与手动上下文切换一致。
                    let old = self.threads[previous].ctx.as_mut_ptr();
                    let new = self.threads[next].ctx.as_ptr();
                    unsafe { switch_context(&mut *old, &*new) };
                }
                return;
            }
        }
        // 没有别的就绪线程：什么都不做，调用方（主循环或 yield 的线程）继续跑。
    }
}

impl TaskContext {
    fn as_mut_ptr(&mut self) -> *mut TaskContext {
        self as *mut TaskContext
    }
    fn as_ptr(&self) -> *const TaskContext {
        self as *const TaskContext
    }
}

static mut SCHEDULER: *mut Scheduler = std::ptr::null_mut();

/// Current thread voluntarily yields; the scheduler will pick the next ready thread.
pub fn yield_now() {
    unsafe {
        if !SCHEDULER.is_null() {
            (*SCHEDULER).schedule_next();
        }
    }
}

/// Mark current thread as `Finished` and switch to the next (called by `thread_wrapper` after the user entry returns).
fn thread_finished() {
    unsafe {
        if !SCHEDULER.is_null() {
            let sched = &mut *SCHEDULER;
            sched.threads[sched.current].state = ThreadState::Finished;
            sched.schedule_next();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// Tests must run serially: the scheduler uses global state (SCHEDULER, CURRENT_THREAD_ENTRY).
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    static EXEC_ORDER: AtomicU32 = AtomicU32::new(0);

    extern "C" fn task_a() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(100, Ordering::SeqCst);
    }

    extern "C" fn task_b() {
        EXEC_ORDER.fetch_add(1, Ordering::SeqCst);
        yield_now();
        EXEC_ORDER.fetch_add(10, Ordering::SeqCst);
    }

    #[test]
    fn test_scheduler_runs_all() {
        let _guard = TEST_LOCK.lock().unwrap();
        EXEC_ORDER.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(task_a);
        sched.spawn(task_b);
        sched.run();

        let got = EXEC_ORDER.load(Ordering::SeqCst);
        if got != 122 {
            panic!(
                "EXEC_ORDER: expected 122, got {} (run with --nocapture to see stderr)",
                got
            );
        }
    }

    static SIMPLE_FLAG: AtomicU32 = AtomicU32::new(0);

    extern "C" fn simple_task() {
        SIMPLE_FLAG.store(42, Ordering::SeqCst);
    }

    #[test]
    fn test_single_thread() {
        let _guard = TEST_LOCK.lock().unwrap();
        SIMPLE_FLAG.store(0, Ordering::SeqCst);

        let mut sched = Scheduler::new();
        sched.spawn(simple_task);
        sched.run();

        assert_eq!(SIMPLE_FLAG.load(Ordering::SeqCst), 42);
    }
}
