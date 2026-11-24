// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use thiserror_ext::AsReport as _;
use tokio::sync::{broadcast, watch};
use tokio_util::task::AbortOnDropHandle;
use tracing::warn;
use uuid::Uuid;
use vllm_engine_core_client::kv_events::{KvEventSubscriber, decode_batch, replay};

use super::{PublisherAddresses, batch_to_proto, pb};

/// Live batches a subscriber may fall behind by before it is caught up from
/// the engine's replay buffer.
const LIVE_CAPACITY: usize = 1024;

/// Pause after a failed read from the publisher, which fails repeatedly while
/// the connection is down.
const RETRY_DELAY: Duration = Duration::from_secs(1);

/// Why a reader cannot start or continue.
#[derive(Debug)]
pub(super) enum ReadError {
    /// The cursor's batches are no longer buffered by the engine, or its
    /// stream is not the current one.
    OutOfRange,
    /// Batches after the last one read can no longer be recovered.
    DataLoss,
    /// The engine's replay socket did not answer.
    ReplayFailed(String),
    /// The relay was dropped with its service.
    RelayStopped,
}

/// Forwards one engine's KV event batches, as its PUB socket delivers them,
/// to any number of readers. It keeps no history: readers fill gaps and catch
/// up from the engine's own replay buffer.
pub(super) struct KvEventRelay {
    live: broadcast::Sender<Arc<pb::KvEventBatch>>,
    stream_id: Arc<str>,
    /// The sequence number after the last batch forwarded.
    forwarded: watch::Receiver<Option<u64>>,
    replay_endpoint: String,
    _task: AbortOnDropHandle<()>,
}

impl KvEventRelay {
    pub async fn connect(publisher: &PublisherAddresses) -> vllm_engine_core_client::Result<Self> {
        let subscriber = KvEventSubscriber::connect(&publisher.endpoint, &publisher.topic).await?;
        let (live, _) = broadcast::channel(LIVE_CAPACITY);
        let (forwarded_sender, forwarded) = watch::channel(None);
        let task = tokio::spawn(follow(subscriber, live.clone(), forwarded_sender));
        Ok(Self {
            live,
            stream_id: Uuid::new_v4().to_string().into(),
            forwarded,
            replay_endpoint: publisher.replay_endpoint.clone(),
            _task: AbortOnDropHandle::new(task),
        })
    }

    /// A reader of live batches, first replaying the engine's buffered batches
    /// from `resume` when given.
    pub async fn reader(
        &self,
        resume: Option<pb::KvEventCursor>,
    ) -> Result<KvEventReader, ReadError> {
        // Subscribe before replaying, so no batch published meanwhile is missed.
        let live = self.live.subscribe();
        let mut reader = KvEventReader {
            live,
            replay_endpoint: self.replay_endpoint.clone(),
            stream_id: self.stream_id.clone(),
            // Live from the next batch the relay forwards.
            next: *self.forwarded.borrow(),
            forwarded: self.forwarded.clone(),
            pending: VecDeque::new(),
        };
        if let Some(cursor) = resume {
            if !cursor.stream_id.is_empty() && *cursor.stream_id != *self.stream_id {
                return Err(ReadError::OutOfRange);
            }
            reader.next = Some(cursor.next_sequence_number);
            reader.catch_up(cursor.next_sequence_number, ReadError::OutOfRange).await?;
        }
        Ok(reader)
    }
}

async fn follow(
    mut subscriber: KvEventSubscriber,
    live: broadcast::Sender<Arc<pb::KvEventBatch>>,
    forwarded: watch::Sender<Option<u64>>,
) {
    loop {
        let (sequence_number, payload) = match subscriber.next().await {
            Ok(published) => published,
            Err(error) => {
                warn!(error = %error.as_report(), "failed to read KV events from the engine");
                tokio::time::sleep(RETRY_DELAY).await;
                continue;
            }
        };
        // A batch that cannot be decoded is a gap that readers cannot fill.
        let Some(batch) = decode_batch(sequence_number, &payload) else {
            continue;
        };
        forwarded.send_replace(Some(sequence_number + 1));
        // No receiver is not an error: readers come and go.
        let _ = live.send(Arc::new(batch_to_proto(sequence_number, batch)));
    }
}

/// One subscriber's position in a relay's stream.
pub(super) struct KvEventReader {
    live: broadcast::Receiver<Arc<pb::KvEventBatch>>,
    replay_endpoint: String,
    stream_id: Arc<str>,
    /// Sequence number of the next batch to yield; `None` until the relay
    /// forwards its first batch.
    next: Option<u64>,
    forwarded: watch::Receiver<Option<u64>>,
    /// Replayed batches to yield before live ones.
    pending: VecDeque<Arc<pb::KvEventBatch>>,
}

impl KvEventReader {
    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    /// The next batch, each sequence number once and in order.
    pub async fn next_batch(&mut self) -> Result<Arc<pb::KvEventBatch>, ReadError> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                self.next = Some(batch.sequence_number + 1);
                return Ok(batch);
            }
            match self.live.recv().await {
                Ok(batch) => match self.next {
                    // Already yielded from a replay.
                    Some(next) if batch.sequence_number < next => continue,
                    // The PUB socket dropped batches; the replay holds this one too.
                    Some(next) if batch.sequence_number > next => {}
                    _ => {
                        self.next = Some(batch.sequence_number + 1);
                        return Ok(batch);
                    }
                },
                // This reader fell more than `LIVE_CAPACITY` batches behind.
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => return Err(ReadError::RelayStopped),
            }
            // A reader that fell behind before its first batch has no position to refill from.
            let next = self.next.ok_or(ReadError::DataLoss)?;
            self.catch_up(next, ReadError::DataLoss).await?;
        }
    }

    /// Queue the engine's buffered batches from `next` on. `missing` is the
    /// error when the buffer does not hold every batch published from `next`.
    async fn catch_up(&mut self, next: u64, missing: ReadError) -> Result<(), ReadError> {
        // Read before the replay: a batch published after its answer is not missing from it.
        let published = self.forwarded.borrow().is_some_and(|forwarded| next < forwarded);
        let batches = replay(&self.replay_endpoint, next)
            .await
            .map_err(|error| ReadError::ReplayFailed(error.to_report_string()))?;
        // An engine with `buffer_steps` 0 buffers nothing.
        if batches.is_empty() && published {
            return Err(missing);
        }
        for (expected, (sequence_number, payload)) in (next..).zip(batches) {
            // The buffer has moved past `next`, or the engine failed to encode a batch.
            if sequence_number != expected {
                return Err(missing);
            }
            let Some(batch) = decode_batch(sequence_number, &payload) else {
                return Err(missing);
            };
            self.pending.push_back(Arc::new(batch_to_proto(sequence_number, batch)));
        }
        Ok(())
    }
}
