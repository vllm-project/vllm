// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! KV cache events of `vllm/distributed/kv_events.py` produced by the block pool.

use crate::hash::BlockHash;

pub const MEDIUM_GPU: &str = "GPU";
pub const MEDIUM_CPU: &str = "CPU";
pub const MEDIUM_STORAGE: &str = "STORAGE";

/// `ExternalBlockHash`: the raw hash bytes, or their low 64 bits as an
/// integer when `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` is set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExternalBlockHash {
    Bytes(BlockHash),
    Int(u64),
}

impl ExternalBlockHash {
    /// `maybe_convert_block_hash`.
    pub fn convert(hash: &BlockHash, use_int_block_hashes: bool) -> Self {
        if !use_int_block_hashes {
            return Self::Bytes(hash.clone());
        }
        // `int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)`.
        let bytes = hash.as_bytes();
        let low = &bytes[bytes.len().saturating_sub(8)..];
        Self::Int(low.iter().fold(0u64, |acc, &b| (acc << 8) | u64::from(b)))
    }
}

/// One untagged extra key of a block as published in KV events
/// (`to_event_extra_keys`): bare LoRA names and cache salts,
/// `(mm_identifier, offset)` pairs and bare prompt-embeds digests.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EventExtraKey {
    /// A LoRA name or a cache salt.
    Str(String),
    /// A prompt-embeds digest.
    Bytes(Vec<u8>),
    /// A multi-modal input: its identifier and its offset relative to the
    /// block start.
    Mm { identifier: String, offset: i64 },
}

/// The extra keys of one block in event format.
pub type EventExtraKeys = Vec<EventExtraKey>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockStored {
    pub block_hashes: Vec<ExternalBlockHash>,
    pub parent_block_hash: Option<ExternalBlockHash>,
    pub token_ids: Vec<i32>,
    pub block_size: usize,
    /// Deprecated: use `lora_name` for KV block key hash.
    pub lora_id: Option<i64>,
    pub medium: Option<String>,
    pub lora_name: Option<String>,
    pub extra_keys: Option<Vec<Option<EventExtraKeys>>>,
    pub group_idx: Option<usize>,
    pub kv_cache_spec_kind: Option<&'static str>,
    pub kv_cache_spec_sliding_window: Option<usize>,
    pub session_id: Option<String>,
    pub lookahead_token_ids: Option<Vec<Vec<i32>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRemoved {
    pub block_hashes: Vec<ExternalBlockHash>,
    pub medium: Option<String>,
    pub group_idx: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllBlocksCleared;

/// Base class for all KV cache-related events.
// Events are mostly `BlockStored`, so boxing it would not save memory.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvCacheEvent {
    BlockStored(BlockStored),
    BlockRemoved(BlockRemoved),
    AllBlocksCleared(AllBlocksCleared),
}
