//! # Async Channel
//!
//! In this exercise, you will use `tokio::sync::mpsc` async channels to implement producer-consumer pattern.
//!
//! ## Concepts
//! - `tokio::sync::mpsc::channel` creates bounded async channels
//! - Async `send` and `recv`
//! - Channel closing mechanism (receiver returns None after all senders are dropped)

use tokio::sync::mpsc;

/// Async producer-consumer:
/// - Create a producer task that sends each element from items sequentially
/// - Create a consumer task that receives all elements and collects them into Vec for return
///
/// Hint: Set channel capacity to items.len().max(1)
pub async fn producer_consumer(items: Vec<String>) -> Vec<String> {
    // Capacity max(1) matters: mpsc::channel(0) is illegal and panics on an empty input.
    let (tx, mut rx) = mpsc::channel::<String>(items.len().max(1));

    tokio::spawn(async move {
        for item in items {
            // send().await suspends the producer when the buffer is full — real backpressure.
            if tx.send(item).await.is_err() {
                break; // consumer gone, nothing left to do
            }
        }
        // Dropping tx here is what makes the consumer's recv() eventually return None.
    });

    // The consumer runs as a task too, so the two sides genuinely interleave rather than
    // the main future draining the channel itself.
    let collected = tokio::spawn(async move {
        let mut received = Vec::new();
        while let Some(item) = rx.recv().await {
            received.push(item);
        }
        received
    });

    collected.await.expect("consumer task failed")
}

/// Fan‑in pattern: multiple producers, one consumer.
/// Create `n_producers` producers, each sending `"producer {id}: message"`.
/// Consumer collects all messages, sorts them, and returns.
pub async fn fan_in(n_producers: usize) -> Vec<String> {
    // Bounded capacity 1 forces every producer to await between messages, so the run really
    // exercises concurrent send contention instead of just filling a roomy buffer.
    let (tx, mut rx) = mpsc::channel::<String>(1);

    for id in 0..n_producers {
        let tx = tx.clone();
        tokio::spawn(async move {
            let _ = tx.send(format!("producer {id}: message")).await;
        });
    }
    // The template sender must go, otherwise the receiver waits on it forever after the clones
    // have all finished.
    drop(tx);

    let mut messages = Vec::with_capacity(n_producers);
    while let Some(msg) = rx.recv().await {
        messages.push(msg);
    }
    messages.sort();
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_producer_consumer() {
        let items = vec!["hello".into(), "async".into(), "world".into()];
        let result = producer_consumer(items.clone()).await;
        assert_eq!(result, items);
    }

    #[tokio::test]
    async fn test_producer_consumer_empty() {
        let result = producer_consumer(vec![]).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_fan_in() {
        let result = fan_in(3).await;
        assert_eq!(
            result,
            vec![
                "producer 0: message",
                "producer 1: message",
                "producer 2: message",
            ]
        );
    }

    #[tokio::test]
    async fn test_fan_in_single() {
        let result = fan_in(1).await;
        assert_eq!(result, vec!["producer 0: message"]);
    }
}
