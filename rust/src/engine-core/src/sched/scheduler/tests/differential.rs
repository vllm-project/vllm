// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Replays scheduler traces recorded from the Python implementation
//! (`gen_traces.py`) and checks that every `SchedulerOutput`,
//! `EngineCoreOutputs`, KV connector call, KV cache usage and request count
//! matches.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::*;
use crate::hash::{BlockHash, HashAlgo, RequestBlockHasher};
use crate::kv_cache::{
    BoundaryStateOffload, KvCacheBlocks, KvCacheGroupSpec, KvCacheSpec, KvCacheSpecType,
};
use crate::sched::output::Opaque;
use crate::sched::request::{NewRequest, SamplingParams, StopReason};

/// Connector calls, recorded on the Rust side in the trace's format.
type CallLog = Arc<Mutex<Vec<Value>>>;

/// Queued `get_num_new_matched_tokens` answers.
type MatchAnswers = Arc<Mutex<VecDeque<(Option<usize>, bool)>>>;

/// Replays `RustCoreDiffConnector` (`tests/v1/core/rust_core_diff.py`): the
/// matched-token answers come from the trace, the rest of its behavior is
/// the base class defaults.
struct ReplayConnector {
    matches: MatchAnswers,
    log: CallLog,
    loaded_group_ids: Vec<usize>,
    requires_kv_delivery: bool,
}

impl KvConnector for ReplayConnector {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        false
    }

    fn requires_kv_delivery(&self) -> bool {
        self.requires_kv_delivery
    }

    fn on_new_request(&mut self, _request: &Request, _kv_cache_manager: &mut KvCacheManager) {}

    fn may_load_async(&mut self, _request: &Request) -> bool {
        // The base default.
        true
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        _kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        let (n, load_async) =
            self.matches.lock().unwrap().pop_front().expect("unexpected connector match");
        self.log.lock().unwrap().push(json!([
            "match",
            request.request_id,
            num_computed_tokens,
            n,
            load_async
        ]));
        (n, load_async)
    }

    fn get_loaded_kv_cache_group_ids(&mut self, _request: &Request) -> Vec<usize> {
        self.loaded_group_ids.clone()
    }

    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        _kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    ) {
        self.log.lock().unwrap().push(json!([
            "alloc",
            request.request_id,
            num_external_tokens,
            blocks.get_block_ids(false, None).unwrap()
        ]));
    }

    fn get_external_cache_hit_sources(
        &mut self,
        _request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        Some(vec![(
            crate::sched::stats::CACHE_HIT_SOURCE_EXTERNAL_UNSPECIFIED.to_owned(),
            num_external_tokens,
        )])
    }

    fn has_pending_block_frees(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        false
    }

    fn build_connector_meta(
        &mut self,
        _scheduler_output: &SchedulerOutput,
        _block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        Opaque::new(())
    }

    fn register_finished_partial_tail(
        &mut self,
        _request: &Request,
        _block_ids: &[Vec<BlockId>],
        _partial_tail_offloads: &[BoundaryStateOffload],
        _kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        false
    }

    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        _kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        self.log
            .lock()
            .unwrap()
            .push(json!(["finished", request.request_id, block_ids]));
        RequestFinished::default()
    }

    fn update_connector_output(
        &mut self,
        _kv_connector_output: &KvConnectorOutput,
        _kv_cache_manager: &mut KvCacheManager,
    ) {
    }

    fn get_kv_connector_stats(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        None
    }

    fn take_events(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        None
    }

    fn has_pending_push_work(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        false
    }

    fn reset_cache(&mut self, _kv_cache_manager: &mut KvCacheManager) -> Option<bool> {
        None
    }
}

pub(super) struct TraceHasher(RequestBlockHasher);

impl crate::sched::request::BlockHasher for TraceHasher {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
        self.0.new_block_hashes(
            request.all_token_ids(),
            request.block_hashes.len(),
            request.block_hashes.last(),
            &Default::default(),
        )
    }
}

pub(super) fn usize_of(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

pub(super) fn mamba_cache_mode(v: &Value) -> MambaCacheMode {
    match v.as_str().unwrap() {
        "none" => MambaCacheMode::None,
        "all" => MambaCacheMode::All,
        "align" => MambaCacheMode::Align,
        other => panic!("unknown mamba cache mode {other}"),
    }
}

pub(super) fn group_spec(g: &Value) -> KvCacheGroupSpec {
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

struct Replay {
    scheduler: Scheduler,
    hasher: Arc<dyn crate::sched::request::BlockHasher>,
    matches: MatchAnswers,
    log: CallLog,
}

/// The scheduler config (without a connector), KV cache config and block
/// hasher of a trace's config.
pub(super) fn trace_setup(
    c: &Value,
) -> (
    SchedulerConfig,
    KvCacheConfig,
    Arc<dyn crate::sched::request::BlockHasher>,
) {
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

fn build(config: &Value) -> Replay {
    let c = config;
    let (mut sc, kv_cache_config, hasher) = trace_setup(c);
    let matches = Arc::new(Mutex::new(VecDeque::new()));
    let log: CallLog = Arc::new(Mutex::new(Vec::new()));
    let connector: Option<Box<dyn KvConnector>> = if c["connector"].is_null() {
        None
    } else {
        sc.kv_transfer = Some(KvTransferConfig {
            is_kv_producer: true,
            is_kv_consumer: true,
            recompute_kv_load_failures: c["recompute_kv_load_failures"].as_bool().unwrap(),
        });
        Some(Box::new(ReplayConnector {
            matches: matches.clone(),
            log: log.clone(),
            loaded_group_ids: c["prefix_cacheable_group_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(usize_of)
                .collect(),
            requires_kv_delivery: c["requires_kv_delivery"].as_bool().unwrap(),
        }))
    };

    let mut scheduler = Scheduler::new(
        sc,
        kv_cache_config,
        connector,
        Box::new(AcceptAllStructuredOutput),
    )
    .unwrap();
    scheduler.set_clock(Box::new(|| 0.0));
    Replay {
        scheduler,
        hasher,
        matches,
        log,
    }
}

pub(super) fn so_json(so: &SchedulerOutput) -> Value {
    let cached = &so.scheduled_cached_reqs;
    let sorted = |ids: &indexmap::IndexSet<String>| {
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

pub(super) fn eco_json(update: &UpdateOutput) -> Value {
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

pub(super) fn floats_match(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= f64::EPSILON * a.abs().max(1.0),
        _ => false,
    }
}

/// The path of the first difference between two JSON values.
pub(super) fn first_diff(a: &Value, b: &Value, path: &str) -> Option<String> {
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

fn replay(path: &Path) {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let name = path.file_name().unwrap().to_string_lossy();
    let Replay {
        mut scheduler,
        hasher,
        matches,
        log,
    } = build(&lines[0]["config"]);
    let mut outputs: Vec<SchedulerOutput> = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(1) {
        let ctx = format!("{name} line {}", i + 1);
        // Preload the connector's answers for the calls this op makes.
        if let Some(next) = lines.get(i + 1)
            && next["op"] == "conn"
        {
            let mut queue = matches.lock().unwrap();
            for call in next["calls"].as_array().unwrap() {
                if call[0] == "match" {
                    queue.push_back((
                        call[3].as_u64().map(|n| n as usize),
                        call[4].as_bool().unwrap(),
                    ));
                }
            }
        }
        match line["op"].as_str().unwrap() {
            "add" => {
                let request = Request::new(NewRequest {
                    request_id: line["id"].as_str().unwrap().to_owned(),
                    prompt_token_ids: Some(
                        line["prompt"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|t| t.as_i64().unwrap() as i32)
                            .collect(),
                    ),
                    sampling_params: Some(SamplingParams {
                        max_tokens: Some(usize_of(&line["max_tokens"])),
                        min_tokens: usize_of(&line["min_tokens"]),
                        eos_token_id: line["eos"].as_i64().map(|t| t as i32),
                        stop_token_ids: line["stop"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|t| t.as_i64().unwrap() as i32)
                            .collect(),
                        ..Default::default()
                    }),
                    arrival_time: Some(line["arrival_time"].as_f64().unwrap()),
                    priority: line["priority"].as_i64().unwrap(),
                    block_hasher: Some(hasher.clone()),
                    ..Default::default()
                })
                .unwrap();
                scheduler.add_request(request);
            }
            "abort" => {
                let ids: Vec<String> = line["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_owned())
                    .collect();
                let result = scheduler.finish_requests(Some(&ids), RequestStatus::FinishedAborted);
                assert_eq!(json!(result), line["result"], "{ctx}: finish_requests");
            }
            "schedule" => {
                let so = scheduler.schedule(false);
                if let Some(d) = first_diff(&line["out"], &so_json(&so), "SchedulerOutput") {
                    panic!("{ctx}: {d}");
                }
                outputs.push(so);
            }
            "update" => {
                let so = &outputs[usize_of(&line["sched_step"])];
                let mro = ModelRunnerOutput {
                    req_ids: line["req_ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                        .collect(),
                    sampled_token_ids: line["sampled"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|t| {
                            t.as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_i64().unwrap() as i32)
                                .collect()
                        })
                        .collect(),
                    kv_connector_output: line["finished_recving"].as_array().map(|ids| {
                        KvConnectorOutput {
                            finished_recving: Some(
                                ids.iter().map(|v| v.as_str().unwrap().to_owned()).collect(),
                            ),
                            ..Default::default()
                        }
                    }),
                    ..Default::default()
                };
                let update = scheduler.update_from_output(so, &mro);
                if let Some(d) = first_diff(&line["eco"], &eco_json(&update), "EngineCoreOutputs") {
                    panic!("{ctx}: {d}");
                }
            }
            "draft" => {
                scheduler.update_draft_token_ids(DraftTokenIds {
                    req_ids: line["req_ids"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                        .collect(),
                    draft_token_ids: line["drafts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|t| {
                            t.as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_i64().unwrap() as i32)
                                .collect()
                        })
                        .collect(),
                });
            }
            "conn" => {
                let calls = std::mem::take(&mut *log.lock().unwrap());
                if let Some(d) = first_diff(&line["calls"], &json!(calls), "connector calls") {
                    panic!("{ctx}: {d}");
                }
                assert!(
                    matches.lock().unwrap().is_empty(),
                    "{ctx}: unused connector answers"
                );
                continue;
            }
            "check" => {
                let usage = scheduler.get_kv_cache_usage();
                assert!(
                    floats_match(Some(usage), line["usage"].as_f64()),
                    "{ctx}: usage {usage}"
                );
                let (r, w) = scheduler.get_request_counts();
                assert_eq!(json!([r, w]), line["counts"], "{ctx}: counts");
            }
            other => panic!("unknown op {other}"),
        }
        // Every connector call must be accounted for by the next `conn` op.
        let next_is_conn = lines.get(i + 1).is_some_and(|n| n["op"] == "conn");
        if !next_is_conn {
            let calls = log.lock().unwrap();
            assert!(
                calls.is_empty(),
                "{ctx}: unexpected connector calls {calls:?}"
            );
        }
    }
}

#[test]
fn matches_python_traces() {
    crate::trace_replay::replay_all("sched", "src/sched/scheduler/tests/traces", "jsonl", replay);
}
