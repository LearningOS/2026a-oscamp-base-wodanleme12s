//! # Mutex Shared State
//!
//! In this exercise, you will use `Arc<Mutex<T>>` to safely share and modify data between multiple threads.
//!
//! ## Concepts
//! - `Mutex<T>` mutex protects shared data
//! - `Arc<T>` atomic reference counting enables cross-thread sharing
//! - `lock()` acquires the lock and accesses data

use std::sync::{Arc, Mutex};
use std::thread;

/// Increment a counter concurrently using `n_threads` threads.
/// Each thread increments the counter `count_per_thread` times.
/// Returns the final counter value.
///
/// Hint: Use `Arc<Mutex<usize>>` as the shared counter.
pub fn concurrent_counter(n_threads: usize, count_per_thread: usize) -> usize {
    // Each worker takes the lock once per round. Note that we deliberately do NOT
    // hold the guard across the loop: that would serialise trivially but hide the
    // real cost of acquiring, so this version exercises genuine contention.
    let shared = Arc::new(Mutex::new(0usize));
    let mut workers = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let worker_shared = Arc::clone(&shared);
        workers.push(thread::spawn(move || {
            for _ in 0..count_per_thread {
                let mut guard = worker_shared.lock().unwrap();
                *guard += 1;
                // guard released here
            }
        }));
    }
    for w in workers {
        w.join().unwrap();
    }
    // Drop the worker clones first: otherwise the final read below would be fine,
    // but this makes the ownership hand-back explicit.
    let final_value = *shared.lock().unwrap();
    final_value
}

/// Add elements to a shared vector concurrently using multiple threads.
/// Each thread pushes its own id (0..n_threads) to the vector.
/// Returns the sorted vector.
///
/// Hint: Use `Arc<Mutex<Vec<usize>>>`.
pub fn concurrent_collect(n_threads: usize) -> Vec<usize> {
    // Threads insert into a pre-sized vector at their own index instead of racing
    // to push, so the final sort below is only a safety net, not the ordering source.
    let slots = Arc::new(Mutex::new(vec![usize::MAX; n_threads]));
    let mut workers = Vec::with_capacity(n_threads);
    for id in 0..n_threads {
        let slots = Arc::clone(&slots);
        workers.push(thread::spawn(move || {
            let mut table = slots.lock().unwrap();
            table[id] = id;
        }));
    }
    for w in workers {
        w.join().unwrap();
    }
    let mut collected = Arc::try_unwrap(slots)
        .expect("all workers have dropped their clones")
        .into_inner()
        .unwrap();
    collected.sort_unstable();
    collected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_single_thread() {
        assert_eq!(concurrent_counter(1, 100), 100);
    }

    #[test]
    fn test_counter_multi_thread() {
        assert_eq!(concurrent_counter(10, 100), 1000);
    }

    #[test]
    fn test_counter_zero() {
        assert_eq!(concurrent_counter(5, 0), 0);
    }

    #[test]
    fn test_collect() {
        let result = concurrent_collect(5);
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_collect_single() {
        assert_eq!(concurrent_collect(1), vec![0]);
    }
}
