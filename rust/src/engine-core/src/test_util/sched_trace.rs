// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! What the replays of scheduler traces recorded from the Python
//! implementation share: the scheduler and KV cache configs of a trace, and
//! the normalized JSON of the outputs they compare.

use std::sync::Arc;

use indexmap::IndexSet;
use serde_json::{Value, json};

use crate::hash::{BlockHash, HashAlgo, RequestBlockHasher, RequestExtraKeys};
use crate::kv_cache::{
    KvCacheConfig, KvCacheGroupSpec, KvCacheSpec, KvCacheSpecType, MambaCacheMode,
};
use crate::sched::output::SchedulerOutput;
use crate::sched::request::{BlockHasher, Request, StopReason};
use crate::sched::request_queue::SchedulingPolicy;
use crate::sched::scheduler::{SchedulerClass, SchedulerConfig, UpdateOutput};

/// A block hasher with `HashAlgo::Sha256` and a zero `NONE_HASH`, without
/// extra keys.
pub fn hasher(block_size: usize) -> Arc<dyn BlockHasher> {
    Arc::new(TraceHasher(RequestBlockHasher::new(
        HashAlgo::Sha256,
        BlockHash::from_slice(&[0; 32]),
        block_size,
        0,
    )))
}

pub struct TraceHasher(RequestBlockHasher);

impl BlockHasher for TraceHasher {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
        self.0.new_block_hashes(
            request.all_token_ids(),
            request.block_hashes.len(),
            request.block_hashes.last(),
            &RequestExtraKeys::default(),
        )
    }
}

pub fn usize_of(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

pub fn mamba_cache_mode(v: &Value) -> MambaCacheMode {
    match v.as_str().unwrap() {
        "none" => MambaCacheMode::None,
        "all" => MambaCacheMode::All,
        "align" => MambaCacheMode::Align,
        other => panic!("unknown mamba cache mode {other}"),
    }
}

pub fn group_spec(g: &Value) -> KvCacheGroupSpec {
    let block_size = usize_of(&g["block_size"]);
    let sliding_window = || usize_of(&g["sliding_window"]);
    let extra_retained_tokens = usize_of(&g["extra_retained_tokens"]);
    let spec_type = match g["type"].as_str().unwrap() {
        "FullAttentionSpec" => KvCacheSpecType::FullAttention,
        "MLAAttentionSpec" => KvCacheSpecType::MlaAttention,
        "SlidingWindowSpec" => KvCacheSpecType::SlidingWindow {
            sliding_window: sliding_window(),
            extra_retained_tokens,
        },
        "SlidingWindowMLASpec" => KvCacheSpecType::SlidingWindowMla {
            sliding_window: sliding_window(),
            extra_retained_tokens,
            bounded_replay: g["bounded_replay"].as_bool().unwrap(),
        },
        "CircularBufferSpec" => KvCacheSpecType::CircularBuffer,
        "MambaSpec" => KvCacheSpecType::Mamba {
            mamba_cache_mode: mamba_cache_mode(&g["mamba_cache_mode"]),
            num_speculative_blocks: usize_of(&g["num_speculative_blocks"]),
            num_prefill_checkpoint_blocks: usize_of(&g["num_prefill_checkpoint_blocks"]),
            prefill_checkpoint_alignment: g["prefill_checkpoint_alignment"]
                .as_u64()
                .map(|n| n as usize),
        },
        other => panic!("unsupported spec {other}"),
    };
    KvCacheGroupSpec::new(KvCacheSpec::new(spec_type, block_size))
}

/// The scheduler config (without a connector), KV cache config and block
/// hasher of a trace's config.
pub fn trace_setup(c: &Value) -> (SchedulerConfig, KvCacheConfig, Arc<dyn BlockHasher>) {
    let block_size = usize_of(&c["block_size"]);
    let mut sc = SchedulerConfig::for_test(block_size, usize_of(&c["max_model_len"]));
    sc.num_gpu_blocks = usize_of(&c["num_blocks"]);
    sc.hash_block_size = Some(usize_of(&c["hash_block_size"]));
    sc.enable_prefix_caching = c["enable_prefix_caching"].as_bool().unwrap();
    sc.max_num_seqs = usize_of(&c["max_num_seqs"]);
    sc.max_num_batched_tokens = usize_of(&c["max_num_batched_tokens"]);
    sc.max_num_scheduled_tokens = Some(usize_of(&c["max_num_scheduled_tokens"]));
    sc.long_prefill_token_threshold = usize_of(&c["long_prefill_token_threshold"]);
    sc.policy = SchedulingPolicy::from_name(c["policy"].as_str().unwrap()).unwrap();
    sc.num_spec_tokens = usize_of(&c["num_spec_tokens"]);
    sc.num_lookahead_tokens = usize_of(&c["num_lookahead_tokens"]);
    sc.num_prefill_lookahead_tokens = usize_of(&c["num_prefill_lookahead"]);
    sc.num_hash_lookahead_tokens = usize_of(&c["num_hash_lookahead_tokens"]);
    sc.use_eagle_block_drop = c["use_eagle_block_drop"].as_bool().unwrap();
    sc.draft_slots = usize_of(&c["draft_slots"]);
    assert!(!c["dynamic_sd"].as_bool().unwrap());
    sc.enable_chunked_prefill = c["enable_chunked_prefill"].as_bool().unwrap();
    sc.watermark = c["watermark"].as_f64().unwrap();
    sc.scheduler_reserve_full_isl = c["scheduler_reserve_full_isl"].as_bool().unwrap();
    sc.cache_block_size = usize_of(&c["cache_block_size"]);
    sc.mamba_cache_mode = mamba_cache_mode(&c["mamba_cache_mode"]);
    sc.enable_mamba_shared_prefix_checkpoint =
        c["enable_mamba_shared_prefix_checkpoint"].as_bool().unwrap();
    assert_eq!(usize_of(&c["max_num_active_reqs"]), sc.max_num_seqs);
    sc.max_in_flight_tokens = c["max_in_flight_tokens"].as_u64().map(|n| n as usize);
    sc.max_concurrent_batches = usize_of(&c["max_concurrent_batches"]);
    sc.async_scheduling = c["async_scheduling"].as_bool().unwrap();
    sc.enable_kv_cache_events = c["kv_events"].as_bool().unwrap();
    sc.log_stats = true;
    if c["async"].as_bool().unwrap() {
        sc.scheduler_class = SchedulerClass::Async { decode_stagger: 1 };
    }

    let groups = c["groups"].as_array().unwrap().iter().map(group_spec).collect();
    let mut kv_cache_config = KvCacheConfig::new(sc.num_gpu_blocks, groups);
    kv_cache_config.prefix_cache_retention_interval =
        c["retention_interval"].as_u64().map(|n| n as usize);
    kv_cache_config.needs_kv_cache_zeroing = c["needs_kv_cache_zeroing"].as_bool().unwrap();

    let none_hash = c["none_hash"].as_str().unwrap();
    let none_hash: Vec<u8> = (0..none_hash.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&none_hash[i..i + 2], 16).unwrap())
        .collect();
    let algo = HashAlgo::from_name(c["hash_fn"].as_str().unwrap()).unwrap();
    let hasher = Arc::new(TraceHasher(RequestBlockHasher::new(
        algo,
        BlockHash::from_slice(&none_hash),
        sc.hash_block_size.unwrap(),
        sc.num_hash_lookahead_tokens,
    )));

    (sc, kv_cache_config, hasher)
}

pub fn so_json(so: &SchedulerOutput) -> Value {
    let cached = &so.scheduled_cached_reqs;
    let sorted = |ids: &IndexSet<String>| {
        let mut v: Vec<&String> = ids.iter().collect();
        v.sort();
        json!(v)
    };
    json!({
        "new": so.scheduled_new_reqs.iter().map(|r| json!([
            r.req_id,
            r.num_computed_tokens,
            r.block_ids,
            r.prefill_token_ids.len(),
            r.prefill_token_ids.iter().map(|&t| t as i64).sum::<i64>(),
            r.replay_start,
        ])).collect::<Vec<_>>(),
        "cached": [
            cached.req_ids,
            cached.new_token_ids,
            cached.new_block_ids,
            cached.num_computed_tokens,
            cached.num_output_tokens,
        ],
        "nst": so.num_scheduled_tokens.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        "total": so.total_num_scheduled_tokens,
        "spec": so.scheduled_spec_decode_tokens.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        "finished": sorted(&so.finished_req_ids),
        "preempted": so.preempted_req_ids.as_ref().map(sorted),
        "zero": so.new_block_ids_to_zero,
        "copies": so.kv_cache_block_copies.as_ref().map(|copies| {
            copies.iter().map(|c| json!([c.src_block_id, c.dst_block_id])).collect::<Vec<_>>()
        }),
        "k": so.num_spec_tokens_to_schedule,
        "skip_drafts": so.skip_drafts,
        "sync_kv": so.has_sync_kv_loads,
    })
}

pub fn eco_json(update: &UpdateOutput) -> Value {
    let clients: Vec<Value> = update
        .engine_core_outputs
        .iter()
        .map(|(client, eco)| {
            let outputs: Vec<Value> = eco
                .outputs
                .iter()
                .map(|o| {
                    let stop_reason = match &o.stop_reason {
                        None => Value::Null,
                        Some(StopReason::TokenId(t)) => json!(t),
                        Some(StopReason::Text(s)) => json!(s),
                    };
                    json!([
                        o.request_id,
                        o.new_token_ids,
                        o.finish_reason.map(|r| r as u8),
                        stop_reason,
                        o.num_nans_in_logits,
                        o.prefill_stats.map(|p| json!([
                            p.num_prompt_tokens,
                            p.num_computed_tokens,
                            p.num_cached_tokens,
                            p.num_local_cached_tokens,
                            p.num_external_cached_tokens,
                            p.num_cache_creation_tokens,
                        ])),
                    ])
                })
                .collect();
            let stats = eco.scheduler_stats.as_ref().map(|s| {
                let p = &s.prefix_cache_stats;
                json!([
                    s.num_running_reqs,
                    s.num_waiting_reqs,
                    s.num_skipped_waiting_reqs,
                    s.kv_cache_usage,
                    [
                        p.reset,
                        p.requests,
                        p.queries,
                        p.hits,
                        p.preempted_requests,
                        p.preempted_queries,
                        p.preempted_hits
                    ],
                    s.spec_decoding_stats.as_ref().map(|d| json!([
                        d.num_drafts,
                        d.num_draft_tokens,
                        d.num_accepted_tokens,
                        d.num_accepted_tokens_per_pos,
                        d.num_draft_tokens_per_pos,
                    ])),
                ])
            });
            let finished = eco.finished_requests.as_ref().map(|f| {
                let mut v: Vec<&String> = f.iter().collect();
                v.sort();
                v
            });
            json!([client, {"outputs": outputs, "finished": finished, "stats": stats}])
        })
        .collect();
    json!(clients)
}

pub fn floats_match(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= f64::EPSILON * a.abs().max(1.0),
        _ => false,
    }
}

/// The path of the first difference between two JSON values.
pub fn first_diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in x {
                let Some(w) = y.get(k) else {
                    return Some(format!("{path}.{k}: missing"));
                };
                if let Some(d) = first_diff(v, w, &format!("{path}.{k}")) {
                    return Some(d);
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .find_map(|(i, (v, w))| first_diff(v, w, &format!("{path}[{i}]"))),
        // Python floats and ints compare by value. serde_json parses floats
        // to within an ulp, not exactly (without `float_roundtrip`).
        (Value::Number(x), Value::Number(y)) if floats_match(x.as_f64(), y.as_f64()) => None,
        _ if a == b => None,
        _ => {
            let (a, b) = (a.to_string(), b.to_string());
            Some(format!(
                "{path}: python {} != rust {}",
                &a[..a.len().min(600)],
                &b[..b.len().min(600)]
            ))
        }
    }
}
