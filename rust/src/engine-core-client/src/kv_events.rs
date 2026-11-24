// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Ordered subscription to one engine's KV-event publisher.

use std::collections::VecDeque;
use std::time::Duration;

use bytes::Bytes;
use thiserror_ext::AsReport as _;
use tracing::warn;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSocket, SocketOptions, SubSocket, ZmqMessage};

use crate::error::{
    Error, Result, bail_unexpected_kv_event_message, bail_unsupported_kv_event_endpoint,
};
use crate::protocol::kv_events::KvEventBatch;

/// How long connecting to one of the publisher's sockets may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the publisher may take to send the next frame of a replay before
/// the missing batches count as lost.
const REPLAY_TIMEOUT: Duration = Duration::from_secs(5);
/// Sequence number of the frame that ends a replay (`ZmqEventPublisher.END_SEQ`).
const REPLAY_END_SEQUENCE_NUMBER: u64 = u64::MAX;

/// A batch with its position in the publisher's stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SequencedKvEventBatch {
    /// The publisher's batch counter, starting at 0.
    pub sequence_number: u64,
    /// Whether batches between the previously yielded one and this one are
    /// missing.
    pub follows_gap: bool,
    pub batch: KvEventBatch,
}

/// Subscriber to the `ZmqEventPublisher` of one engine.
///
/// Yields each batch once, in sequence-number order, starting with the first
/// one received. Batches the PUB socket dropped are fetched from the
/// publisher's replay socket when it has one.
// TODO: a batch dropped right before the publisher goes idle stays unnoticed
// until the next one arrives; probe the replay socket while idle.
pub struct KvEventSubscriber {
    events: SubSocket,
    /// Address of the publisher's replay socket, if it has one.
    replay_endpoint: Option<String>,
    /// Sequence number after the last accepted batch; `None` before the first.
    next_sequence_number: Option<u64>,
    /// Whether a batch was lost since the last yielded one.
    gap: bool,
    /// Replayed batches not yet yielded.
    replayed: VecDeque<(u64, Bytes)>,
}

impl KvEventSubscriber {
    /// Connect to a publisher's events address and, if it has one, its replay
    /// address.
    pub async fn connect(
        endpoint: &str,
        replay_endpoint: Option<&str>,
        topic: &str,
    ) -> Result<Self> {
        let mut events = SubSocket::with_options(socket_options());
        events.connect(&connect_endpoint(endpoint)?).await?;
        events.subscribe(topic).await?;

        Ok(Self {
            events,
            replay_endpoint: replay_endpoint.map(connect_endpoint).transpose()?,
            next_sequence_number: None,
            gap: false,
            replayed: VecDeque::new(),
        })
    }

    /// The next batch in sequence-number order.
    pub async fn next(&mut self) -> Result<SequencedKvEventBatch> {
        loop {
            while let Some((sequence_number, payload)) = self.replayed.pop_front() {
                if let Some(batch) = self.accept(sequence_number, &payload) {
                    return Ok(batch);
                }
            }

            let (sequence_number, payload) = published_batch(self.events.recv().await?, 0)?;
            if let Some(next) = self.next_sequence_number
                && sequence_number > next
                && let Some(endpoint) = &self.replay_endpoint
            {
                match replay(endpoint, next).await {
                    Ok(batches) => self.replayed.extend(batches),
                    Err(error) => {
                        warn!(start = next, error = %error.as_report(), "KV event replay failed")
                    }
                }
            }
            // Behind anything replayed, which covers it unless the replay failed.
            self.replayed.push_back((sequence_number, payload));
        }
    }

    /// Decode a batch unless it was already accepted, recording what it
    /// reveals as lost.
    fn accept(&mut self, sequence_number: u64, payload: &[u8]) -> Option<SequencedKvEventBatch> {
        match self.next_sequence_number {
            // A replay overlapping what the PUB socket already delivered. The
            // publisher lives as long as its engine, which outlives this
            // client, so its counter never restarts.
            Some(next) if sequence_number < next => return None,
            Some(next) if sequence_number > next => self.gap = true,
            _ => {}
        }
        self.next_sequence_number = Some(sequence_number + 1);

        match decode_batch(sequence_number, payload) {
            Some(batch) => Some(SequencedKvEventBatch {
                sequence_number,
                follows_gap: std::mem::take(&mut self.gap),
                batch,
            }),
            None => {
                self.gap = true;
                None
            }
        }
    }
}

/// Decode a published batch, logging one that cannot be decoded.
pub fn decode_batch(sequence_number: u64, payload: &[u8]) -> Option<KvEventBatch> {
    KvEventBatch::decode(payload)
        .inspect_err(|error| {
            warn!(sequence_number, error = %error.as_report(), "dropping undecodable KV event batch");
        })
        .ok()
}

/// The publisher's buffered batches from `start` on, in order: `(sequence
/// number, payload)`. Batches older than its buffer are not included.
pub async fn replay(endpoint: &str, start: u64) -> Result<Vec<(u64, Bytes)>> {
    // One socket per replay: the late reply to a replay that timed out is
    // never read as the next one's, and a connection lost while idle needs no
    // repair.
    let mut socket = DealerSocket::with_options(socket_options());
    socket.connect(endpoint).await?;
    // The empty frame is the REQ envelope the publisher's ROUTER expects.
    let request = vec![Bytes::new(), Bytes::copy_from_slice(&start.to_be_bytes())];
    socket
        .send(ZmqMessage::try_from(request).expect("replay request has frames"))
        .await?;

    let mut batches = Vec::new();
    loop {
        let message =
            tokio::time::timeout(REPLAY_TIMEOUT, socket.recv()).await.map_err(|_| {
                Error::KvEventReplayTimeout {
                    timeout: REPLAY_TIMEOUT,
                }
            })??;
        let (sequence_number, payload) = published_batch(message, 1)?;
        if sequence_number == REPLAY_END_SEQUENCE_NUMBER {
            return Ok(batches);
        }
        batches.push((sequence_number, payload));
    }
}

fn socket_options() -> SocketOptions {
    let mut options = SocketOptions::default();
    options.connect_timeout(CONNECT_TIMEOUT);
    options
}

/// Split a `(topic, sequence number, payload)` message that follows
/// `envelope_frames` envelope frames.
fn published_batch(message: ZmqMessage, envelope_frames: usize) -> Result<(u64, Bytes)> {
    let mut frames = message.into_vec();
    let sequence_number = match frames.get(envelope_frames + 1).map(|frame| frame.as_ref()) {
        Some(&[b0, b1, b2, b3, b4, b5, b6, b7]) if frames.len() == envelope_frames + 3 => {
            u64::from_be_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
        }
        _ => bail_unexpected_kv_event_message!(
            "expected {} frames with an 8-byte sequence number, got frame lengths {:?}",
            envelope_frames + 3,
            frames.iter().map(Bytes::len).collect::<Vec<_>>()
        ),
    };
    Ok((sequence_number, frames.swap_remove(envelope_frames + 2)))
}

/// The address that reaches a publisher bound to `endpoint`.
pub fn connect_endpoint(endpoint: &str) -> Result<String> {
    if endpoint.starts_with("inproc://") {
        bail_unsupported_kv_event_endpoint!(
            endpoint = endpoint,
            "an inproc endpoint is only reachable inside the engine process"
        );
    }
    let Some((host, port)) =
        endpoint.strip_prefix("tcp://").and_then(|address| address.rsplit_once(':'))
    else {
        return Ok(endpoint.to_string());
    };
    let host = match host {
        "*" | "0.0.0.0" => "127.0.0.1",
        "[::]" => "[::1]",
        host => host,
    };
    Ok(format!("tcp://{host}:{port}"))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::sync::mpsc;
    use tokio_util::task::AbortOnDropHandle;

    use super::{KvEventSubscriber, SequencedKvEventBatch, connect_endpoint};
    use crate::protocol::kv_events::{BlockStored, KvCacheEvent, KvEventBatch};
    use crate::test_utils::{IpcNamespace, MockKvEventPublisher};

    fn batch(token_id: u32) -> KvEventBatch {
        KvEventBatch {
            ts: 0.0,
            events: vec![KvCacheEvent::BlockStored(BlockStored {
                token_ids: vec![token_id],
                ..Default::default()
            })],
            data_parallel_rank: Some(0),
        }
    }

    /// Yield a subscriber's batches as `(sequence number, follows gap)`.
    async fn follow(
        publisher: &MockKvEventPublisher,
    ) -> (mpsc::UnboundedReceiver<(u64, bool)>, AbortOnDropHandle<()>) {
        let config = publisher.config();
        let mut subscriber = KvEventSubscriber::connect(
            config.frontend_endpoint.as_deref().unwrap(),
            config.frontend_replay_endpoint.as_deref(),
            &config.topic,
        )
        .await
        .unwrap();
        let (sender, receiver) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            loop {
                let SequencedKvEventBatch {
                    sequence_number,
                    follows_gap,
                    ..
                } = subscriber.next().await.unwrap();
                let _ = sender.send((sequence_number, follows_gap));
            }
        });
        (receiver, AbortOnDropHandle::new(task))
    }

    /// Publish a batch until the subscriber yields something. A PUB socket
    /// drops what it sends before the subscription reaches it, and the
    /// subscriber ignores the repeats.
    async fn publish_until_received(
        publisher: &mut MockKvEventPublisher,
        sequence_number: u64,
        received: &mut mpsc::UnboundedReceiver<(u64, bool)>,
    ) -> (u64, bool) {
        loop {
            publisher
                .publish_payload(
                    sequence_number,
                    rmp_serde::to_vec_named(&batch(0)).unwrap().into(),
                )
                .await;
            if let Ok(batch) =
                tokio::time::timeout(Duration::from_millis(20), received.recv()).await
            {
                return batch.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn replay_supplies_batches_the_pub_socket_did_not_deliver() {
        let ipc = IpcNamespace::new().unwrap();
        let mut publisher = MockKvEventPublisher::bind(&ipc, true).await;
        // Published before the subscriber existed, so not part of its stream.
        publisher.retain(0, &batch(0));
        let (mut received, _task) = follow(&publisher).await;
        assert_eq!(
            publish_until_received(&mut publisher, 1, &mut received).await,
            (1, false)
        );

        // Dropped by the PUB socket and noticed when the next batch arrives.
        publisher.retain(2, &batch(2));
        publisher.retain(3, &batch(3));
        assert_eq!(
            publish_until_received(&mut publisher, 3, &mut received).await,
            (2, false)
        );
        assert_eq!(received.recv().await, Some((3, false)));

        // And again: every replay stands on its own.
        publisher.retain(4, &batch(4));
        publisher.publish(5, &batch(5)).await;
        assert_eq!(received.recv().await, Some((4, false)));
        assert_eq!(received.recv().await, Some((5, false)));
    }

    #[tokio::test]
    async fn lost_and_undecodable_batches_mark_the_next_one() {
        let ipc = IpcNamespace::new().unwrap();
        let mut publisher = MockKvEventPublisher::bind(&ipc, false).await;
        let (mut received, _task) = follow(&publisher).await;
        // The first batch only sets the starting point.
        assert_eq!(
            publish_until_received(&mut publisher, 5, &mut received).await,
            (5, false)
        );

        publisher.publish(7, &batch(7)).await;
        assert_eq!(received.recv().await, Some((7, true)));
        publisher.publish(8, &batch(8)).await;
        assert_eq!(received.recv().await, Some((8, false)));

        publisher.publish_payload(9, vec![0xc1].into()).await;
        publisher.publish(10, &batch(10)).await;
        assert_eq!(received.recv().await, Some((10, true)));
    }

    #[test]
    fn wildcard_bind_addresses_connect_over_loopback() {
        assert_eq!(
            connect_endpoint("tcp://*:5557").unwrap(),
            "tcp://127.0.0.1:5557"
        );
        assert_eq!(
            connect_endpoint("tcp://0.0.0.0:5557").unwrap(),
            "tcp://127.0.0.1:5557"
        );
        assert_eq!(
            connect_endpoint("tcp://[::]:5557").unwrap(),
            "tcp://[::1]:5557"
        );
        assert_eq!(
            connect_endpoint("tcp://10.0.0.7:5557").unwrap(),
            "tcp://10.0.0.7:5557"
        );
        assert_eq!(connect_endpoint("ipc:///tmp/kv").unwrap(), "ipc:///tmp/kv");
        assert!(connect_endpoint("inproc://kv").is_err());
    }
}
