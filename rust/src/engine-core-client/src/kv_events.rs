// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Subscription to one engine's KV-event publisher and its replay socket.

use std::time::Duration;

use bytes::Bytes;
use thiserror_ext::AsReport as _;
use tracing::warn;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSocket, SocketOptions, SubSocket, ZmqMessage};

use crate::error::{Error, Result, bail_unexpected_kv_event_message};
use crate::protocol::kv_events::KvEventBatch;

/// How long connecting to one of the publisher's sockets may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the publisher may take to send the next frame of a replay before
/// the missing batches count as lost.
const REPLAY_TIMEOUT: Duration = Duration::from_secs(5);
/// Sequence number of the frame that ends a replay (`ZmqEventPublisher.END_SEQ`).
const REPLAY_END_SEQUENCE_NUMBER: u64 = u64::MAX;

/// Subscriber to the PUB socket of one engine's `ZmqEventPublisher`.
pub struct KvEventSubscriber {
    events: SubSocket,
}

impl KvEventSubscriber {
    pub async fn connect(endpoint: &str, topic: &str) -> Result<Self> {
        let mut events = SubSocket::with_options(socket_options());
        events.connect(endpoint).await?;
        events.subscribe(topic).await?;
        Ok(Self { events })
    }

    /// The next batch the PUB socket delivers, with its sequence number. The
    /// socket drops batches its subscriber does not read in time, which the
    /// publisher's replay socket still has.
    pub async fn next(&mut self) -> Result<(u64, Bytes)> {
        published_batch(self.events.recv().await?, 0)
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
