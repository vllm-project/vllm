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

use super::{batch_to_proto, pb};

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

/// The sequence numbers a relay has forwarded.
#[derive(Clone, Copy)]
struct Forwarded {
    first: u64,
    next: u64,
}

#[derive(Clone)]
struct LiveBatch {
    stream_id: Arc<str>,
    batch: Arc<pb::KvEventBatch>,
}

/// Follows one engine's KV-event publisher and fans its batches out to any
/// number of readers. It keeps no history: catch-ups are served from the
/// engine's own replay buffer.
pub(super) struct KvEventRelay {
    live: broadcast::Sender<LiveBatch>,
    stream_id: watch::Receiver<Arc<str>>,
    forwarded: watch::Receiver<Option<Forwarded>>,
    replay_endpoint: Option<String>,
    _task: AbortOnDropHandle<()>,
}

impl KvEventRelay {
    pub async fn connect(
        endpoint: &str,
        replay_endpoint: Option<&str>,
        topic: &str,
    ) -> vllm_engine_core_client::Result<Self> {
        let subscriber = KvEventSubscriber::connect(endpoint, replay_endpoint, topic).await?;
        let replay_endpoint = replay_endpoint
            .map(vllm_engine_core_client::kv_events::connect_endpoint)
            .transpose()?;
        let (live, _) = broadcast::channel(LIVE_CAPACITY);
        let (stream_id_sender, stream_id) = watch::channel(new_stream_id());
        let (forwarded_sender, forwarded) = watch::channel(None);
        let task = tokio::spawn(follow(
            subscriber,
            live.clone(),
            stream_id_sender,
            forwarded_sender,
        ));
        Ok(Self {
            live,
            stream_id,
            forwarded,
            replay_endpoint,
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
            stream_id: self.stream_id.borrow().clone(),
            // Live from the next batch the relay forwards.
            next: self.forwarded.borrow().map(|forwarded| forwarded.next),
            forwarded: self.forwarded.clone(),
            pending: VecDeque::new(),
        };
        if let Some(cursor) = resume {
            if !cursor.stream_id.is_empty() && *cursor.stream_id != *reader.stream_id {
                return Err(ReadError::OutOfRange);
            }
            reader.next = Some(cursor.next_sequence_number);
            reader.catch_up(ReadError::OutOfRange).await?;
        }
        Ok(reader)
    }
}

fn new_stream_id() -> Arc<str> {
    Uuid::new_v4().to_string().into()
}

async fn follow(
    mut subscriber: KvEventSubscriber,
    live: broadcast::Sender<LiveBatch>,
    stream_id: watch::Sender<Arc<str>>,
    forwarded: watch::Sender<Option<Forwarded>>,
) {
    loop {
        match subscriber.next().await {
            Ok(batch) => {
                if batch.follows_gap {
                    warn!(
                        sequence_number = batch.sequence_number,
                        "KV event batches were lost; starting a new stream"
                    );
                    stream_id.send_replace(new_stream_id());
                }
                let sequence_number = batch.sequence_number;
                forwarded.send_modify(|forwarded| {
                    let first = forwarded.map_or(sequence_number, |forwarded| forwarded.first);
                    *forwarded = Some(Forwarded {
                        first,
                        next: sequence_number + 1,
                    });
                });
                let batch = LiveBatch {
                    stream_id: stream_id.borrow().clone(),
                    batch: Arc::new(batch_to_proto(sequence_number, batch.batch)),
                };
                // No receiver is not an error: readers come and go.
                let _ = live.send(batch);
            }
            Err(error) => {
                warn!(error = %error.as_report(), "failed to read KV events from the engine");
                tokio::time::sleep(RETRY_DELAY).await;
            }
        }
    }
}

/// One subscriber's position in a relay's stream.
pub(super) struct KvEventReader {
    live: broadcast::Receiver<LiveBatch>,
    replay_endpoint: Option<String>,
    stream_id: Arc<str>,
    /// Sequence number of the next batch to yield; `None` until the relay
    /// forwards its first batch.
    next: Option<u64>,
    forwarded: watch::Receiver<Option<Forwarded>>,
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
            let item = match self.live.recv().await {
                Ok(item) => item,
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // A reader that started before the relay's first batch
                    // missed everything from that one on.
                    let first = self.forwarded.borrow().map(|forwarded| forwarded.first);
                    self.next = self.next.or(first);
                    self.catch_up(ReadError::DataLoss).await?;
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => return Err(ReadError::RelayStopped),
            };
            if *item.stream_id != *self.stream_id {
                return Err(ReadError::DataLoss);
            }
            let sequence_number = item.batch.sequence_number;
            match self.next {
                Some(next) if sequence_number < next => {}
                Some(next) if sequence_number > next => {
                    self.catch_up(ReadError::DataLoss).await?;
                    if self.pending.back().is_none_or(|last| last.sequence_number < sequence_number)
                    {
                        self.pending.push_back(item.batch);
                    }
                }
                _ => {
                    self.next = Some(sequence_number + 1);
                    return Ok(item.batch);
                }
            }
        }
    }

    /// Queue the engine's buffered batches from `next` on. `missing` is the
    /// error when the buffer no longer starts at `next`.
    async fn catch_up(&mut self, missing: ReadError) -> Result<(), ReadError> {
        let next = self.next.expect("catch-up needs a position");
        let Some(endpoint) = &self.replay_endpoint else {
            return Err(missing);
        };
        let batches = replay(endpoint, next)
            .await
            .map_err(|error| ReadError::ReplayFailed(error.to_report_string()))?;
        if batches.first().is_some_and(|(sequence_number, _)| *sequence_number != next) {
            return Err(missing);
        }
        self.pending.clear();
        for (sequence_number, payload) in batches {
            let Some(batch) = decode_batch(sequence_number, &payload) else {
                return Err(missing);
            };
            self.pending.push_back(Arc::new(batch_to_proto(sequence_number, batch)));
        }
        Ok(())
    }
}
