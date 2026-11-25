// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Subscription to one engine's KV-event publisher.

use std::time::Duration;

use bytes::Bytes;
use zeromq::prelude::{Socket, SocketRecv};
use zeromq::{SocketOptions, SubSocket};

use crate::error::{Error, Result, bail_unexpected_kv_event_message};

/// How long connecting to the publisher may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Subscriber to the PUB socket of one engine's `ZmqEventPublisher`.
pub struct KvEventSubscriber {
    events: SubSocket,
}

impl KvEventSubscriber {
    pub async fn connect(endpoint: &str, topic: &str) -> Result<Self> {
        let mut options = SocketOptions::default();
        options.connect_timeout(CONNECT_TIMEOUT);
        let mut events = SubSocket::with_options(options);
        events.connect(endpoint).await?;
        events.subscribe(topic).await?;
        Ok(Self { events })
    }

    /// The next batch the PUB socket delivers, with its sequence number. The
    /// socket drops batches its subscriber does not read in time.
    pub async fn next(&mut self) -> Result<(u64, Bytes)> {
        // `(topic, sequence number, payload)`.
        let mut frames = self.events.recv().await?.into_vec();
        let sequence_number = match frames.get(1).map(|frame| frame.as_ref()) {
            Some(&[b0, b1, b2, b3, b4, b5, b6, b7]) if frames.len() == 3 => {
                u64::from_be_bytes([b0, b1, b2, b3, b4, b5, b6, b7])
            }
            _ => bail_unexpected_kv_event_message!(
                "expected 3 frames with an 8-byte sequence number, got frame lengths {:?}",
                frames.iter().map(Bytes::len).collect::<Vec<_>>()
            ),
        };
        Ok((sequence_number, frames.swap_remove(2)))
    }
}
