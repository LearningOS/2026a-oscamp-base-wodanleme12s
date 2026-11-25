//! # Tokio Async Tasks
//!
//! In this exercise, you will use `tokio::spawn` to create concurrent asynchronous tasks.
//!
//! ## Concepts
//! - `tokio::spawn` creates asynchronous tasks
//! - `JoinHandle` waits for task completion
//! - Concurrent execution between asynchronous tasks

use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};

/// Concurrently compute the square of each number in 0..n, collect results and return in order.
///
/// Hint: Create `tokio::spawn` task for each i, collect JoinHandle, await them sequentially.
pub async fn concurrent_squares(n: usize) -> Vec<usize> {
    // Spawn everything first, then await in index order: awaiting inside the loop would
    // serialise the tasks and defeat the point of spawn. Handles are pre-sized so results
    // land at their natural index without a later sort.
    let mut handles: Vec<JoinHandle<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        handles.push(tokio::spawn(async move { i * i }));
    }

    let mut squares = Vec::with_capacity(n);
    for handle in handles {
        squares.push(handle.await.expect("square task panicked"));
    }
    squares
}

/// Concurrently execute multiple "time-consuming" tasks (simulated with sleep), return all results.
/// Each task sleeps `duration_ms` milliseconds and then returns its `task_id`.
///
/// Key: All tasks should execute concurrently, total duration should be close to single task duration, not sum of all tasks.
pub async fn parallel_sleep_tasks(n: usize, duration_ms: u64) -> Vec<usize> {
    // sleep() only yields this task, it never blocks the worker thread, so all n tasks make
    // progress concurrently and total wall time stays near a single duration_ms.
    let mut handles: Vec<JoinHandle<usize>> = Vec::with_capacity(n);
    for id in 0..n {
        handles.push(tokio::spawn(async move {
            sleep(Duration::from_millis(duration_ms)).await;
            id
        }));
    }

    let mut finished = Vec::with_capacity(n);
    for handle in handles {
        finished.push(handle.await.expect("sleep task panicked"));
    }
    // Completion order follows the awaited handle order, not wake-up order; sorting keeps the
    // contract explicit even if the collection strategy changes.
    finished.sort_unstable();
    finished
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Instant;

    #[tokio::test]
    async fn test_squares_basic() {
        let result = concurrent_squares(5).await;
        assert_eq!(result, vec![0, 1, 4, 9, 16]);
    }

    #[tokio::test]
    async fn test_squares_zero() {
        let result = concurrent_squares(0).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_squares_one() {
        let result = concurrent_squares(1).await;
        assert_eq!(result, vec![0]);
    }

    #[tokio::test]
    async fn test_parallel_sleep() {
        let start = Instant::now();
        let result = parallel_sleep_tasks(5, 100).await;
        let elapsed = start.elapsed();

        assert_eq!(result, vec![0, 1, 2, 3, 4]);
        // Concurrent execution, total time should be much less than 5 * 100ms
        assert!(
            elapsed.as_millis() < 400,
            "Tasks should run concurrently, took {}ms",
            elapsed.as_millis()
        );
    }
}
