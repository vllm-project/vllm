// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod relay;

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use futures::Stream;
use thiserror_ext::AsReport as _;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};
use vllm_engine_core_client::protocol::handshake::KvEventsConfig;
use vllm_engine_core_client::protocol::kv_events::{
    BlockRemoved, BlockStored, ExternalBlockHash, ExtraKeyValue, KvCacheEvent, KvEventBatch,
};

use self::relay::{KvEventReader, KvEventRelay, ReadError};
use super::{KvEventsServer, pb};
use crate::state::AppState;

pub(crate) type KvEventsGrpcService = KvEventsServer<KvEventsServiceImpl>;

/// gRPC KV event service: re-publishes each engine's ZMQ KV events as typed
/// streams.
pub struct KvEventsServiceImpl {
    /// By data-parallel rank: the private addresses its engine publishes on
    /// for this frontend, and its relay, started by its first subscriber.
    ranks: BTreeMap<u32, (Option<PublisherAddresses>, OnceCell<KvEventRelay>)>,
    /// Ends the streams, which would otherwise hold up a graceful shutdown.
    shutdown: CancellationToken,
}

/// Where an engine publishes its KV events for this frontend.
pub(super) struct PublisherAddresses {
    endpoint: String,
    replay_endpoint: String,
    topic: String,
}

impl PublisherAddresses {
    /// The private addresses of an engine that publishes KV events to this
    /// frontend: its ZMQ publisher reports the two it bound.
    pub(super) fn of(config: Option<&KvEventsConfig>) -> Option<Self> {
        let config =
            config.filter(|config| config.enable_kv_cache_events && config.publisher == "zmq")?;
        Some(Self {
            endpoint: config.frontend_endpoint.clone()?,
            replay_endpoint: config.frontend_replay_endpoint.clone()?,
            topic: config.topic.clone(),
        })
    }
}

impl KvEventsServiceImpl {
    pub fn new(state: Arc<AppState>, shutdown: CancellationToken) -> Self {
        let ranks = (state.engine_core_client().ready_responses().into_iter())
            .map(|ready| {
                let publisher = PublisherAddresses::of(ready.kv_events_config.as_ref());
                (ready.data_parallel_rank, (publisher, OnceCell::new()))
            })
            .collect();
        Self { ranks, shutdown }
    }
}

#[tonic::async_trait]
impl pb::kv_events_server::KvEvents for KvEventsServiceImpl {
    type SubscribeKvEventsStream =
        Pin<Box<dyn Stream<Item = Result<pb::SubscribeKvEventsResponse, Status>> + Send>>;

    async fn subscribe_kv_events(
        &self,
        request: Request<pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        let request = request.into_inner();
        let rank = request.data_parallel_rank;
        let (publisher, relay) = self.ranks.get(&rank).ok_or_else(|| {
            Status::invalid_argument(format!(
                "data parallel rank {rank} is not connected to this frontend; \
                 connected ranks: {:?}",
                self.ranks.keys().collect::<Vec<_>>()
            ))
        })?;
        let publisher = publisher.as_ref().ok_or_else(|| {
            Status::failed_precondition(format!(
                "the engine of rank {rank} does not publish KV cache events; \
                 enable them with --kv-events-config"
            ))
        })?;
        // A failed connect is retried by the next subscriber.
        let relay =
            relay
                .get_or_try_init(|| KvEventRelay::connect(publisher))
                .await
                .map_err(|error| {
                    Status::unavailable(format!(
                        "failed to connect to the KV event publisher of rank {rank}: {}",
                        error.as_report()
                    ))
                })?;

        let reader = relay.reader(request.resume).await.map_err(read_status)?;
        Ok(Response::new(Box::pin(kv_event_responses(
            reader,
            rank,
            self.shutdown.clone(),
        ))))
    }
}

fn read_status(error: ReadError) -> Status {
    match error {
        ReadError::OutOfRange => Status::out_of_range(
            "the cursor's KV event batches are no longer buffered by the engine, or its stream \
             is not the current one; subscribe without `resume` and rebuild",
        ),
        ReadError::DataLoss => Status::data_loss(
            "KV event batches after the last one sent can no longer be recovered; subscribe \
             without `resume` and rebuild",
        ),
        ReadError::ReplayFailed(error) => {
            Status::unavailable(format!("the engine's KV event replay failed: {error}"))
        }
        ReadError::RelayStopped => Status::unavailable("the KV event relay stopped"),
    }
}

#[try_stream]
async fn kv_event_responses(
    mut reader: KvEventReader,
    data_parallel_rank: u32,
    shutdown: CancellationToken,
    mut y: TryYielder<pb::SubscribeKvEventsResponse, Status>,
) -> Result<(), Status> {
    use pb::subscribe_kv_events_response::Item;

    let start = pb::KvEventStreamStart {
        stream_id: reader.stream_id().to_string(),
        data_parallel_rank,
    };
    y.yield_ok(pb::SubscribeKvEventsResponse {
        item: Some(Item::Start(start)),
    })
    .await;

    loop {
        let batch = tokio::select! {
            _ = shutdown.cancelled() => {
                return Err(Status::unavailable("the server is shutting down"));
            }
            batch = reader.next_batch() => batch.map_err(read_status)?,
        };
        y.yield_ok(pb::SubscribeKvEventsResponse {
            item: Some(Item::Batch(Arc::unwrap_or_clone(batch))),
        })
        .await;
    }
}

fn batch_to_proto(sequence_number: u64, batch: KvEventBatch) -> pb::KvEventBatch {
    pb::KvEventBatch {
        sequence_number,
        ts: batch.ts,
        data_parallel_rank: batch.data_parallel_rank,
        events: batch.events.into_iter().map(event_to_proto).collect(),
    }
}

fn event_to_proto(event: KvCacheEvent) -> pb::KvCacheEvent {
    use pb::kv_cache_event::Event;

    let event = match event {
        KvCacheEvent::BlockStored(stored) => Event::BlockStored(stored_to_proto(stored)),
        KvCacheEvent::BlockRemoved(removed) => Event::BlockRemoved(removed_to_proto(removed)),
        KvCacheEvent::AllBlocksCleared => Event::AllBlocksCleared(pb::AllBlocksCleared {}),
    };
    pb::KvCacheEvent { event: Some(event) }
}

fn stored_to_proto(stored: BlockStored) -> pb::BlockStored {
    let token_id_list = |token_ids: Vec<u32>| pb::TokenIdList {
        token_ids: token_ids.into_iter().map(u64::from).collect(),
    };
    let extra_keys =
        stored
            .extra_keys
            .unwrap_or_default()
            .into_iter()
            .map(|keys| pb::BlockExtraKeys {
                keys: keys.unwrap_or_default().into_iter().map(extra_key_to_proto).collect(),
            });

    pb::BlockStored {
        block_hashes: stored.block_hashes.into_iter().map(hash_to_proto).collect(),
        parent_block_hash: stored.parent_block_hash.map(hash_to_proto),
        token_ids: Some(pb::BlockTokenIds {
            ids: Some(pb::block_token_ids::Ids::Single(token_id_list(
                stored.token_ids,
            ))),
        }),
        block_size: stored.block_size,
        lora_id: stored.lora_id,
        medium: stored.medium,
        lora_name: stored.lora_name,
        session_id: stored.session_id,
        cache_salt: None,
        mm_routing_spans: Vec::new(),
        extra_keys: extra_keys.collect(),
        group_idx: stored.group_idx,
        kv_cache_spec_kind: stored.kv_cache_spec_kind,
        kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
        locality: stored.locality,
        ownership: stored.ownership,
        lookahead_token_ids: (stored.lookahead_token_ids.unwrap_or_default().into_iter())
            .map(token_id_list)
            .collect(),
    }
}

fn removed_to_proto(removed: BlockRemoved) -> pb::BlockRemoved {
    pb::BlockRemoved {
        block_hashes: removed.block_hashes.into_iter().map(hash_to_proto).collect(),
        medium: removed.medium,
        group_idx: removed.group_idx,
        locality: removed.locality,
        ownership: removed.ownership,
    }
}

fn hash_to_proto(hash: ExternalBlockHash) -> pb::BlockHash {
    use pb::block_hash::Value;

    let value = match hash {
        ExternalBlockHash::Unsigned(value) => Value::UnsignedValue(value),
        ExternalBlockHash::Signed(value) => Value::SignedValue(value),
        ExternalBlockHash::Digest(digest) => Value::Digest(digest),
    };
    pb::BlockHash { value: Some(value) }
}

fn extra_key_to_proto(key: ExtraKeyValue) -> pb::ExtraKeyValue {
    use pb::extra_key_value::Value;

    let value = match key {
        ExtraKeyValue::Text(text) => Value::Text(text),
        ExtraKeyValue::Data(data) => Value::Data(data),
        ExtraKeyValue::Integer(value) => Value::Integer(value),
        ExtraKeyValue::Tuple(items) => Value::Tuple(pb::ExtraKeyTuple {
            items: items.into_iter().map(extra_key_to_proto).collect(),
        }),
    };
    pb::ExtraKeyValue { value: Some(value) }
}
