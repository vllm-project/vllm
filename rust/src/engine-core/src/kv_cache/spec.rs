// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The parts of `vllm/v1/kv_cache_interface.py` the KV cache manager reads at
//! runtime.
//!
//! Spec classes become [`KvCacheSpecType`] variants carrying only the fields
//! the managers read. Python compares specs as dataclasses (class and every
//! field except `is_draft`) when batching groups for cache-hit lookup, so the
//! fields this port drops (head sizes, dtypes, page padding, ...) are folded
//! into [`KvCacheSpec::fingerprint`] by whoever builds the config.
//!
//! `UniformTypeKVCacheSpecs` is not represented: the scheduler-side config
//! (`generate_scheduler_kv_cache_config`) replaces it with its first layer
//! spec, which the binding layer must do too.

use super::cdiv;

/// `MambaSpec.mamba_cache_mode` (`CacheConfig.mamba_cache_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MambaCacheMode {
    None,
    All,
    Align,
}

/// The spec class of a KV cache group, with the class-specific fields the
/// managers read.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KvCacheSpecType {
    /// `FullAttentionSpec`.
    FullAttention,
    /// `MLAAttentionSpec` (a `FullAttentionSpec`).
    MlaAttention,
    /// `HiddenStateCacheSpec` (an `MLAAttentionSpec`).
    HiddenStateCache,
    /// `RSWASpec` (a `FullAttentionSpec`).
    Rswa { rswa_window: usize },
    /// `SinkFullAttentionSpec` (a `FullAttentionSpec`).
    SinkFullAttention { sink_len: Option<usize> },
    /// `SlidingWindowSpec`.
    SlidingWindow {
        sliding_window: usize,
        extra_retained_tokens: usize,
    },
    /// `SlidingWindowMLASpec` (a `SlidingWindowSpec`).
    SlidingWindowMla {
        sliding_window: usize,
        extra_retained_tokens: usize,
        bounded_replay: bool,
    },
    /// `ChunkedLocalAttentionSpec`.
    ChunkedLocalAttention { attention_chunk_size: usize },
    /// `CircularBufferSpec`.
    CircularBuffer,
    /// `MambaSpec`.
    Mamba {
        mamba_cache_mode: MambaCacheMode,
        num_speculative_blocks: usize,
        num_prefill_checkpoint_blocks: usize,
        prefill_checkpoint_alignment: Option<usize>,
    },
    /// `CrossAttentionSpec`.
    CrossAttention,
    /// `EncoderOnlyAttentionSpec`. Has no registered manager.
    EncoderOnlyAttention,
    /// `HiSparseHotSpec`: ephemeral per-request HiSparse hot-cache allocation.
    HiSparseHot { blocks_per_request: usize },
    /// `HiSparseResidentSpec`: reclaimable GPU-resident pages for host-backed
    /// HiSparse KV.
    HiSparseResident,
}

/// A base class for specifying the KV cache format of one layer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheSpec {
    /// number of tokens in a block
    pub block_size: usize,
    /// Whether DCP shards this cache's token positions across ranks.
    /// (`AttentionSpec` defaults this to True, other specs to False.)
    pub dcp_sharded: bool,
    pub spec_type: KvCacheSpecType,
    /// Opaque fingerprint of the Python spec fields this port does not model,
    /// so that spec equality matches Python dataclass equality. Specs that
    /// Python considers equal must carry equal fingerprints.
    pub fingerprint: u64,
}

impl KvCacheSpec {
    /// A spec with the given type and block size, `dcp_sharded` defaulted as
    /// in Python and a zero fingerprint.
    pub fn new(spec_type: KvCacheSpecType, block_size: usize) -> Self {
        let dcp_sharded = spec_type.is_attention();
        Self {
            block_size,
            dcp_sharded,
            spec_type,
            fingerprint: 0,
        }
    }

    /// `isinstance(spec, FullAttentionSpec)`.
    pub fn is_full_attention(&self) -> bool {
        self.spec_type.is_full_attention()
    }

    /// `isinstance(spec, SlidingWindowSpec)`.
    pub fn is_sliding_window(&self) -> bool {
        matches!(
            self.spec_type,
            KvCacheSpecType::SlidingWindow { .. } | KvCacheSpecType::SlidingWindowMla { .. }
        )
    }

    /// `isinstance(spec, MambaSpec)`.
    pub fn is_mamba(&self) -> bool {
        matches!(self.spec_type, KvCacheSpecType::Mamba { .. })
    }

    /// `isinstance(spec, AttentionSpec)`.
    pub fn is_attention(&self) -> bool {
        self.spec_type.is_attention()
    }

    /// `isinstance(spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec))`.
    pub fn is_cross_or_encoder_only(&self) -> bool {
        matches!(
            self.spec_type,
            KvCacheSpecType::CrossAttention | KvCacheSpecType::EncoderOnlyAttention
        )
    }

    /// The sliding window of a `SlidingWindowSpec`.
    pub fn sliding_window(&self) -> Option<usize> {
        match self.spec_type {
            KvCacheSpecType::SlidingWindow { sliding_window, .. }
            | KvCacheSpecType::SlidingWindowMla { sliding_window, .. } => Some(sliding_window),
            _ => None,
        }
    }

    /// The mamba cache mode of a `MambaSpec`.
    pub fn mamba_cache_mode(&self) -> Option<MambaCacheMode> {
        match self.spec_type {
            KvCacheSpecType::Mamba {
                mamba_cache_mode, ..
            } => Some(mamba_cache_mode),
            _ => None,
        }
    }

    /// Whether this spec's group participates in prefix caching.
    pub fn prefix_cacheable(&self) -> bool {
        match self.spec_type {
            KvCacheSpecType::HiSparseHot { .. }
            | KvCacheSpecType::HiSparseResident
            | KvCacheSpecType::CircularBuffer => false,
            KvCacheSpecType::SlidingWindowMla { bounded_replay, .. } => !bounded_replay,
            _ => true,
        }
    }

    /// DeepSeek-V4.1 only: bounded replay. Currently only for DSV41 SWA.
    pub fn prefix_replay_tokens(&self) -> usize {
        match self.spec_type {
            KvCacheSpecType::SlidingWindowMla {
                sliding_window,
                bounded_replay: true,
                ..
            } => sliding_window,
            _ => 0,
        }
    }

    /// Per-request admission cap, in blocks, for the spec types that recycle
    /// blocks (`SlidingWindowSpec` / `ChunkedLocalAttentionSpec`
    /// `.max_admission_blocks_per_request`); `None` for the others.
    pub fn max_admission_blocks_per_request(
        &self,
        max_in_flight_tokens: usize,
        max_model_len: usize,
    ) -> Option<usize> {
        match self.spec_type {
            KvCacheSpecType::SlidingWindow {
                sliding_window,
                extra_retained_tokens,
            }
            | KvCacheSpecType::SlidingWindowMla {
                sliding_window,
                extra_retained_tokens,
                ..
            } => {
                // During chunked prefill, we hold KV for the last `sliding_window-1`
                // computed tokens plus the in-flight tokens (frees happen on the
                // processed-token basis); never more than `max_model_len`. An additional
                // `extra_retained_tokens` trailing tokens are kept alive below the
                // window for multi-module spec decoding, and must be accounted here too.
                let num_tokens =
                    (sliding_window - 1 + extra_retained_tokens + max_in_flight_tokens)
                        .min(max_model_len);
                // +1 because the sliding window may not start from the beginning of
                // the block. E.g. block size 4 and num_token 4 needs two blocks
                // [XXCD][EF] to store the 6-token window [CDEF].
                Some(cdiv(num_tokens, self.block_size) + 1)
            }
            KvCacheSpecType::ChunkedLocalAttention {
                attention_chunk_size,
            } => {
                // During chunked prefill, we hold KV for at most one chunk window plus
                // the in-flight tokens, since frees happen on the processed-token basis.
                let num_tokens = (attention_chunk_size + max_in_flight_tokens).min(max_model_len);
                Some(cdiv(num_tokens, self.block_size))
            }
            KvCacheSpecType::HiSparseResident => {
                let num_tokens = max_in_flight_tokens.min(max_model_len);
                Some(cdiv(num_tokens, self.block_size))
            }
            _ => None,
        }
    }

    /// `get_kv_cache_spec_kind(spec).value`, reported in `BlockStored` events.
    pub fn kind_str(&self) -> &'static str {
        match self.spec_type {
            KvCacheSpecType::SlidingWindowMla { .. } => "sliding_window_mla",
            KvCacheSpecType::MlaAttention | KvCacheSpecType::HiddenStateCache => "mla_attention",
            KvCacheSpecType::SinkFullAttention { .. } => "sink_full_attention",
            KvCacheSpecType::FullAttention | KvCacheSpecType::Rswa { .. } => "full_attention",
            KvCacheSpecType::ChunkedLocalAttention { .. } => "chunked_local_attention",
            KvCacheSpecType::SlidingWindow { .. } => "sliding_window",
            KvCacheSpecType::Mamba { .. } => "mamba",
            KvCacheSpecType::EncoderOnlyAttention => "encoder_only_attention",
            KvCacheSpecType::CrossAttention => "cross_attention",
            KvCacheSpecType::CircularBuffer
            | KvCacheSpecType::HiSparseHot { .. }
            | KvCacheSpecType::HiSparseResident => "unknown",
        }
    }
}

impl KvCacheSpecType {
    /// `isinstance(spec, FullAttentionSpec)`.
    pub fn is_full_attention(&self) -> bool {
        matches!(
            self,
            Self::FullAttention
                | Self::MlaAttention
                | Self::HiddenStateCache
                | Self::Rswa { .. }
                | Self::SinkFullAttention { .. }
        )
    }

    /// `isinstance(spec, AttentionSpec)`.
    pub fn is_attention(&self) -> bool {
        !matches!(
            self,
            Self::Mamba { .. } | Self::HiSparseHot { .. } | Self::HiSparseResident
        )
    }
}

/// `KVCacheGroupRole`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvCacheGroupRole {
    #[default]
    Default,
    HiSparseSource,
    HiSparseIndexer,
}

/// Represents a group of model layers that share the same KV cache block table.
/// These layers are regarded as one layer in the KV cache manager.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheGroupSpec {
    /// The KV cache spec of this manager layer
    pub kv_cache_spec: KvCacheSpec,
    /// Whether this group holds draft layers and takes the EAGLE last-block drop.
    pub is_eagle_group: bool,
    /// Host groups use the dedicated HiSparse pool; others share the device pool.
    pub host_resident: bool,
    /// Whether this group is part of the externally transferable KV state.
    pub enable_kv_transfer: bool,
    pub role: KvCacheGroupRole,
}

impl KvCacheGroupSpec {
    pub fn new(kv_cache_spec: KvCacheSpec) -> Self {
        Self {
            kv_cache_spec,
            is_eagle_group: false,
            host_resident: false,
            enable_kv_transfer: true,
            role: KvCacheGroupRole::Default,
        }
    }
}

/// The KV cache configuration of a model (the scheduler-side projection).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// The number of KV cache blocks
    pub num_blocks: usize,
    /// The kv cache groups of the model.
    pub kv_cache_groups: Vec<KvCacheGroupSpec>,
    /// Resolved retention policy for local prefix-cache checkpoints.
    pub prefix_cache_retention_interval: Option<usize>,
    /// Whether newly allocated attention blocks must be zeroed before use.
    /// Python derives it from the layer dtypes, which this port does not model.
    pub needs_kv_cache_zeroing: bool,
    /// Capacity of the dedicated HiSparse host-block manager, when enabled.
    pub hisparse_host_num_blocks: Option<usize>,
}

impl KvCacheConfig {
    pub fn new(num_blocks: usize, kv_cache_groups: Vec<KvCacheGroupSpec>) -> Self {
        let mut config = Self {
            num_blocks,
            kv_cache_groups,
            prefix_cache_retention_interval: None,
            needs_kv_cache_zeroing: false,
            hisparse_host_num_blocks: None,
        };
        // `has_mamba_layers or has_mixed_precision_kv_cache`; precision is not
        // modeled, so callers set the flag explicitly for mixed precision.
        config.needs_kv_cache_zeroing = config.has_mamba_layers();
        config
    }

    pub fn has_mamba_layers(&self) -> bool {
        self.kv_cache_groups.iter().any(|g| g.kv_cache_spec.is_mamba())
    }
}

/// Return the reusable Mamba checkpoint boundary for a prefill.
pub fn get_mamba_prefill_checkpoint_position(
    num_tokens: usize,
    hash_block_size: usize,
    drop_eagle_block: bool,
    num_hash_lookahead_tokens: usize,
) -> usize {
    // Leave tokens for logits recomputation and the checkpoint's hash.
    // EAGLE's block drop already leaves tokens to recompute.
    let proof_limit = if drop_eagle_block {
        num_tokens as i64
    } else {
        num_tokens as i64 - num_hash_lookahead_tokens.max(1) as i64
    };
    let hbs = hash_block_size as i64;
    let mut checkpoint_position = proof_limit.div_euclid(hbs) * hbs;
    if drop_eagle_block {
        checkpoint_position -= hbs;
    }
    checkpoint_position.max(0) as usize
}

/// Whether a backend can export the checkpoint in this query.
pub fn is_mamba_prefill_checkpoint_valid(
    query_start: usize,
    query_end: usize,
    checkpoint_position: usize,
    hash_block_size: usize,
    mamba_block_size: usize,
    checkpoint_alignment: Option<usize>,
) -> bool {
    let Some(checkpoint_alignment) = checkpoint_alignment else {
        return false;
    };
    assert!(checkpoint_alignment > 0);

    // Python floor division: `(0 - 1) // n == -1`.
    let initial_state_col = (query_start as i64 - 1).div_euclid(mamba_block_size as i64);
    let checkpoint_col = cdiv(query_end, mamba_block_size) as i64 - 2;
    query_start.is_multiple_of(hash_block_size)
        && checkpoint_col > initial_state_col
        && query_start + hash_block_size <= checkpoint_position
        && query_start < checkpoint_position
        && checkpoint_position < query_end
        && (checkpoint_position - query_start).is_multiple_of(checkpoint_alignment)
}
