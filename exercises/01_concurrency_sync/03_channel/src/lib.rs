//! # Channel Communication
//!
//! In this exercise, you will use `std::sync::mpsc` channels to pass messages between threads.
//!
//! ## Concepts
//! - `mpsc::channel()` creates a multiple producer, single consumer channel
//! - `Sender::send()` sends a message
//! - `Receiver::recv()` receives a message
//! - Multiple producers can be created via `Sender::clone()`

use std::sync::mpsc;
use std::thread;

/// Create a producer thread that sends each element from items into the channel.
/// The main thread receives all messages and returns them.
pub fn simple_send_recv(items: Vec<String>) -> Vec<String> {
    // The producer owns the only Sender; when the spawned closure finishes, that
    // Sender drops and the consumer's blocking recv() starts returning Err, which
    // is what terminates the drain loop below.
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        for item in items {
            // A send failure only means the consumer already went away.
            if tx.send(item).is_err() {
                break;
            }
        }
    });

    let mut received = Vec::new();
    while let Ok(item) = rx.recv() {
        received.push(item);
    }
    received
}

/// Create `n_producers` producer threads, each sending a message in format `"msg from {id}"`.
/// Collect all messages, sort them lexicographically, and return.
///
/// Hint: Use `tx.clone()` to create multiple senders. Note that the original tx must also be dropped.
pub fn multi_producer(n_producers: usize) -> Vec<String> {
    // Every producer gets its own Sender clone; the parent's copy is dropped
    // explicitly, otherwise rx.iter() would block forever on the extra reference.
    let (tx, rx) = mpsc::channel::<String>();
    let mut workers = Vec::with_capacity(n_producers);
    for id in 0..n_producers {
        let tx = tx.clone();
        workers.push(thread::spawn(move || {
            let _ = tx.send(format!("msg from {id}"));
        }));
    }
    drop(tx);

    for w in workers {
        w.join().unwrap();
    }
    // iter() ends once the last Sender is gone, so we get exactly n_producers msgs.
    let mut messages: Vec<String> = rx.iter().collect();
    messages.sort();
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_send_recv() {
        let items = vec!["hello".into(), "world".into(), "rust".into()];
        let result = simple_send_recv(items.clone());
        assert_eq!(result, items);
    }

    #[test]
    fn test_simple_empty() {
        let result = simple_send_recv(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_multi_producer() {
        let result = multi_producer(3);
        assert_eq!(
            result,
            vec![
                "msg from 0".to_string(),
                "msg from 1".to_string(),
                "msg from 2".to_string(),
            ]
        );
    }

    #[test]
    fn test_multi_producer_single() {
        let result = multi_producer(1);
        assert_eq!(result, vec!["msg from 0".to_string()]);
    }
}
