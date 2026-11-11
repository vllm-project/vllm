// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Replays the `MooncakeStoreConnector` traces recorded from the Python
//! scheduler and connector (`gen_mooncake_traces.py`) against the Rust
//! scheduler with the native connector
//! ([`crate::mooncake_store_scheduler`]), and checks every
//! `SchedulerOutput`, connector metadata, `EngineCoreOutputs`, block pool
//! state (ref counts and free queue order), store-job pins and lookup client
//! call.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use indexmap::IndexSet;
use serde_json::{Value, json};

use crate::mooncake_store::{LookupResult, TailKeyBoundary};
use crate::mooncake_store_scheduler::{
    KvRole, LookupClient, MooncakeStoreConnectorMetadata, MooncakeStoreScheduler,
    MooncakeStoreSchedulerConfig, MooncakeStoreWorkerMetadata, ReqBlockHashes, SavePinPolicy,
    SetOp, StoreGeometry,
};
use vllm_engine_core::kv_cache::KvCacheManager;
use vllm_engine_core::kv_cache::{BlockId, BoundaryStateOffload, KvCacheBlocks};
use vllm_engine_core::sched::output::Opaque;
use vllm_engine_core::sched::output::SchedulerOutput;
use vllm_engine_core::sched::request::{NewRequest, SamplingParams};
use vllm_engine_core::sched::request::{Request, RequestStatus};
use vllm_engine_core::sched::scheduler::{
    AcceptAllStructuredOutput, DraftTokenIds, KvConnector, KvConnectorBlockState,
    KvConnectorOutput, KvTransferConfig, ModelRunnerOutput, RequestFinished, Scheduler,
};
use vllm_engine_core::test_util::sched_trace::{
    eco_json, first_diff, floats_match, so_json, trace_setup, usize_of,
};
use vllm_engine_core::test_util::trace_replay::replay_all;

/// The lookup client calls an op is expected to make, in order.
type ExpectedCalls = Arc<Mutex<VecDeque<Value>>>;

/// Answers lookups as the Python run's client did, checking each call.
struct ScriptedClient(ExpectedCalls);

impl ScriptedClient {
    fn expect(&self, call: Value) -> Value {
        let expected = self.0.lock().unwrap().pop_front();
        let Some(expected) = expected else {
            panic!("unexpected lookup client call {call}");
        };
        let n = call.as_array().unwrap().len();
        assert_eq!(
            expected.as_array().unwrap()[..n],
            call.as_array().unwrap()[..],
            "lookup client call"
        );
        expected
    }
}

impl LookupClient for ScriptedClient {
    fn prefetch(&mut self, request: &Request) -> bool {
        let call = json!([
            "prefetch",
            request.request_id,
            request.num_tokens(),
            request.block_hashes.len()
        ]);
        self.expect(call)[4].as_bool().unwrap()
    }

    fn lookup(&mut self, request: &Request, non_block: bool) -> Option<LookupResult> {
        let call = json!([
            "lookup",
            request.request_id,
            request.num_tokens(),
            request.block_hashes.len(),
            non_block
        ]);
        let result = &self.expect(call)[5];
        let result = result.as_array()?;
        Some(LookupResult {
            hit_length: usize_of(&result[0]),
            tail_key_boundaries: result[1]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| TailKeyBoundary {
                    group_id: usize_of(&b[0]),
                    num_tokens: usize_of(&b[1]),
                })
                .collect(),
        })
    }

    fn discard(&mut self, request_id: &str) {
        self.expect(json!(["discard", request_id]));
    }

    fn reset(&mut self) -> bool {
        self.expect(json!(["reset"]));
        true
    }
}

/// The native connector, shared with the test so it can inspect the pins.
/// Records how long each `build_connector_meta` took.
struct SharedConnector(Arc<Mutex<MooncakeStoreScheduler>>, BuildTimes);

type BuildTimes = Arc<Mutex<Vec<std::time::Duration>>>;

impl KvConnector for SharedConnector {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        self.0.lock().unwrap().supports_divergent_local_hybrid_hits()
    }

    fn requires_kv_delivery(&self) -> bool {
        self.0.lock().unwrap().requires_kv_delivery()
    }

    fn on_new_request(&mut self, request: &Request, kv_cache_manager: &mut KvCacheManager) {
        self.0.lock().unwrap().on_new_request(request, kv_cache_manager)
    }

    fn may_load_async(&mut self, request: &Request) -> bool {
        self.0.lock().unwrap().may_load_async(request)
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request: &mut Request,
        num_computed_tokens: usize,
        kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        self.0.lock().unwrap().get_num_new_matched_tokens(
            request,
            num_computed_tokens,
            kv_cache_manager,
        )
    }

    fn get_loaded_kv_cache_group_ids(&mut self, request: &Request) -> Vec<usize> {
        self.0.lock().unwrap().get_loaded_kv_cache_group_ids(request)
    }

    fn update_state_after_alloc(
        &mut self,
        request: &Request,
        blocks: &KvCacheBlocks,
        kv_cache_manager: &mut KvCacheManager,
        num_external_tokens: usize,
    ) {
        self.0.lock().unwrap().update_state_after_alloc(
            request,
            blocks,
            kv_cache_manager,
            num_external_tokens,
        )
    }

    fn get_external_cache_hit_sources(
        &mut self,
        request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        self.0
            .lock()
            .unwrap()
            .get_external_cache_hit_sources(request, num_external_tokens)
    }

    fn has_pending_block_frees(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool {
        self.0.lock().unwrap().has_pending_block_frees(kv_cache_manager)
    }

    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        let mut connector = self.0.lock().unwrap();
        let start = std::time::Instant::now();
        let meta = connector.build_connector_meta(scheduler_output, block_state);
        self.1.lock().unwrap().push(start.elapsed());
        meta
    }

    fn register_finished_partial_tail(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        partial_tail_offloads: &[BoundaryStateOffload],
        kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        self.0.lock().unwrap().register_finished_partial_tail(
            request,
            block_ids,
            partial_tail_offloads,
            kv_cache_manager,
        )
    }

    fn request_finished(
        &mut self,
        request: &Request,
        block_ids: &[Vec<BlockId>],
        kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        self.0.lock().unwrap().request_finished(request, block_ids, kv_cache_manager)
    }

    fn update_connector_output(
        &mut self,
        kv_connector_output: &KvConnectorOutput,
        kv_cache_manager: &mut KvCacheManager,
    ) {
        self.0
            .lock()
            .unwrap()
            .update_connector_output(kv_connector_output, kv_cache_manager)
    }

    fn get_kv_connector_stats(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        self.0.lock().unwrap().get_kv_connector_stats(kv_cache_manager)
    }

    fn take_events(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<Opaque> {
        self.0.lock().unwrap().take_events(kv_cache_manager)
    }

    fn has_pending_push_work(&mut self, kv_cache_manager: &mut KvCacheManager) -> bool {
        self.0.lock().unwrap().has_pending_push_work(kv_cache_manager)
    }

    fn reset_cache(&mut self, kv_cache_manager: &mut KvCacheManager) -> Option<bool> {
        self.0.lock().unwrap().reset_cache(kv_cache_manager)
    }
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(usize_of).collect()
}

/// The native connector's configuration from the trace's `mooncake` config,
/// which the generator read from the constructed Python connector.
fn connector_config(m: &Value) -> MooncakeStoreSchedulerConfig {
    MooncakeStoreSchedulerConfig {
        kv_role: KvRole::from_name(m["kv_role"].as_str().unwrap()).unwrap(),
        load_async: m["load_async"].as_bool().unwrap(),
        lookup_async: m["lookup_async"].as_bool().unwrap(),
        enable_lookup: m["enable_lookup"].as_bool().unwrap(),
        save_decode_cache: m["save_decode_cache"].as_bool().unwrap(),
        store_group_ids: usizes(&m["store_group_ids"]),
        block_size: usize_of(&m["block_size"]),
        hash_block_size: usize_of(&m["hash_block_size"]),
        boundary_state_group_ids: usizes(&m["boundary_state_group_ids"]),
        num_workers: usize_of(&m["num_workers"]),
        store: m["store"].as_object().map(|s| StoreGeometry {
            group_block_sizes: usizes(&s["group_block_sizes"]),
            mamba_group_ids: usizes(&s["mamba_group_ids"]),
            hash_block_size: usize_of(&s["hash_block_size"]),
            lcm_block_size: usize_of(&s["lcm_block_size"]),
            use_eagle: s["use_eagle"].as_bool().unwrap(),
            num_hash_lookahead_tokens: usize_of(&s["num_hash_lookahead_tokens"]),
            eagle_proof_margin_by_group: s["eagle_proof_margin_by_group"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (usize_of(&p[0]), usize_of(&p[1])))
                .collect(),
            enable_partial_hash_hits: s["enable_partial_hash_hits"].as_bool().unwrap(),
        }),
        save_pin_policy: SavePinPolicy::WholeTable,
    }
}

/// FNV-1a over the little-endian `u32`s of `values` (the generator's `fnv`).
fn fnv(values: impl IntoIterator<Item = u32>) -> String {
    fnv_bytes(values.into_iter().flat_map(u32::to_le_bytes))
}

fn fnv_bytes(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        h = (h ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn pool_json(scheduler: &Scheduler) -> Value {
    let pool = scheduler.kv_cache_manager().block_pool();
    json!({
        "refs": fnv(pool.blocks().iter().map(|b| b.ref_cnt as u32)),
        "free": fnv(pool.free_block_queue.get_all_free_blocks()),
    })
}

fn pins_json(connector: &MooncakeStoreScheduler) -> Value {
    let mut pins: Vec<Value> = Vec::new();
    let mut jobs: Vec<_> = connector.pinned_saves().collect();
    jobs.sort_by_key(|&(job, _)| job);
    for (job, pinned) in jobs {
        pins.push(json!([job, pinned.block_ids, pinned.remaining]));
    }
    json!(pins)
}

/// The generator's `canonical_meta`. Python reports the changes to the
/// unfinished set by diffing it, discards sorted.
fn meta_json(meta: &MooncakeStoreConnectorMetadata, unfinished: &mut IndexSet<String>) -> Value {
    let requests: Vec<Value> = meta
        .requests
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let alias = (0..i).find(|&j| Arc::ptr_eq(&meta.requests[j].block_ids, &m.block_ids));
            let hashes = match &m.block_hashes {
                ReqBlockHashes::Request { len } => json!(["live", len]),
                ReqBlockHashes::Finished { hashes, copy } => json!([
                    "finished",
                    copy,
                    hashes.len(),
                    fnv_bytes(hashes.iter().flat_map(|h| h.as_bytes().iter().copied()))
                ]),
            };
            json!({
                "req_id": m.req_id,
                "token_len_chunk": m.token_len_chunk,
                "block_ids": *m.block_ids,
                "alias": alias,
                "hashes": hashes,
                "can_save": m.can_save,
                "load_spec": m.load_spec.as_ref().map(|s| json!([
                    s.vllm_cached_tokens,
                    s.kvpool_cached_tokens,
                    s.can_load,
                    s.token_len,
                    s.tail_key_boundaries
                        .iter()
                        .map(|b| json!([b.group_id, b.num_tokens]))
                        .collect::<Vec<_>>(),
                ])),
                "token_ids": null,
                "token_ids_start": m.token_ids_start,
                "num_hash_lookahead_tokens": 0,
                "num_prompt_tokens": m.num_prompt_tokens,
                "prefill_end_tokens": m.prefill_end_tokens,
                "store_job_id": m.store_job_id,
                "boundary_puts": m.boundary_puts.as_ref().map(|puts| {
                    puts.iter()
                        .map(|p| json!([p.group_id, p.block_id, p.num_tokens]))
                        .collect::<Vec<_>>()
                }),
                "completed_token_len": m.completed_token_len,
                "publish_partial_tail": m.publish_partial_tail,
            })
        })
        .collect();
    let mut ops: Vec<Value> = Vec::new();
    let mut discards: Vec<&String> = Vec::new();
    for op in &meta.unfinished_request_id_ops {
        match op {
            SetOp::Add(id) => {
                assert!(discards.is_empty(), "an add after this step's discards");
                unfinished.insert(id.clone());
                ops.push(json!(["add", id]));
            }
            SetOp::Discard(id) => {
                unfinished.shift_remove(id);
                discards.push(id);
            }
        }
    }
    discards.sort();
    ops.extend(discards.into_iter().map(|id| json!(["discard", id])));
    let mut unfinished_sorted: Vec<&String> = unfinished.iter().collect();
    unfinished_sorted.sort();
    let mut preempted: Vec<&String> = meta.preempted_req_ids.iter().collect();
    preempted.sort();
    json!({
        "requests": requests,
        "unfinished_ops": ops,
        "unfinished": unfinished_sorted,
        "preempted": preempted,
    })
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_owned()).collect()
}

fn token_lists(v: &Value) -> Vec<Vec<i32>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect())
        .collect()
}

fn kv_connector_output(kv: &Value) -> Option<KvConnectorOutput> {
    let kv = kv.as_object()?;
    let worker_meta = kv["completed_saves"].as_array().map(|saves| {
        Opaque::new(MooncakeStoreWorkerMetadata {
            completed_saves: saves
                .iter()
                .map(|s| (s[0].as_u64().unwrap(), usize_of(&s[1])))
                .collect(),
            saved_offsets: Vec::new(),
        })
    });
    Some(KvConnectorOutput {
        finished_recving: kv["finished_recving"]
            .as_array()
            .map(|_| strings(&kv["finished_recving"])),
        failed_recving: strings(&kv["failed_recving"]),
        output: worker_meta,
        ..Default::default()
    })
}

fn replay(path: &Path) -> Vec<std::time::Duration> {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let name = path.file_name().unwrap().to_string_lossy();
    let c = &lines[0]["config"];
    let (mut sc, kv_cache_config, hasher) = trace_setup(c);
    sc.kv_transfer = Some(KvTransferConfig {
        is_kv_producer: c["is_kv_producer"].as_bool().unwrap(),
        is_kv_consumer: c["is_kv_consumer"].as_bool().unwrap(),
        recompute_kv_load_failures: c["recompute_kv_load_failures"].as_bool().unwrap(),
    });
    let expected_calls: ExpectedCalls = Arc::default();
    let build_times = BuildTimes::default();
    let connector = Arc::new(Mutex::new(MooncakeStoreScheduler::new(
        connector_config(&c["mooncake"]),
        Box::new(ScriptedClient(expected_calls.clone())),
    )));
    let mut scheduler = Scheduler::new(
        sc,
        kv_cache_config,
        Some(Box::new(SharedConnector(
            connector.clone(),
            build_times.clone(),
        ))),
        Box::new(AcceptAllStructuredOutput),
    )
    .unwrap();
    scheduler.set_clock(Box::new(|| 0.0));

    let mut unfinished = IndexSet::new();
    let mut outputs: Vec<SchedulerOutput> = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(1) {
        let ctx = format!("{name} line {}", i + 1);
        let diff = |expected: &Value, actual: &Value, what: &str| {
            if let Some(d) = first_diff(expected, actual, what) {
                panic!("{ctx}: {d}");
            }
        };
        let check_state = |scheduler: &Scheduler| {
            if !line["pool"].is_null() {
                diff(&line["pool"], &pool_json(scheduler), "block pool");
            }
            diff(
                &line["pins"],
                &pins_json(&connector.lock().unwrap()),
                "pins",
            );
        };
        *expected_calls.lock().unwrap() =
            line["client"].as_array().cloned().unwrap_or_default().into();
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
                let ids = strings(&line["ids"]);
                let result = scheduler.finish_requests(Some(&ids), RequestStatus::FinishedAborted);
                assert_eq!(json!(result), line["result"], "{ctx}: finish_requests");
                check_state(&scheduler);
            }
            "schedule" => {
                let so = scheduler.schedule(false);
                diff(&line["out"], &so_json(&so), "SchedulerOutput");
                let meta = so
                    .kv_connector_metadata
                    .as_ref()
                    .and_then(|m| m.0.downcast_ref::<MooncakeStoreConnectorMetadata>())
                    .expect("Mooncake store metadata");
                diff(&line["meta"], &meta_json(meta, &mut unfinished), "metadata");
                check_state(&scheduler);
                outputs.push(so);
            }
            "update" => {
                let so = &outputs[usize_of(&line["sched_step"])];
                let mro = ModelRunnerOutput {
                    req_ids: strings(&line["req_ids"]),
                    sampled_token_ids: token_lists(&line["sampled"]),
                    kv_connector_output: kv_connector_output(&line["kv"]),
                    ..Default::default()
                };
                let update = scheduler.update_from_output(so, &mro);
                diff(&line["eco"], &eco_json(&update), "EngineCoreOutputs");
                check_state(&scheduler);
                let push_work = connector
                    .lock()
                    .unwrap()
                    .has_pending_push_work(scheduler.kv_cache_manager_mut());
                assert_eq!(json!(push_work), line["push_work"], "{ctx}: push work");
            }
            "draft" => {
                scheduler.update_draft_token_ids(DraftTokenIds {
                    req_ids: strings(&line["req_ids"]),
                    draft_token_ids: token_lists(&line["drafts"]),
                });
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
        let left = expected_calls.lock().unwrap();
        assert!(
            left.is_empty(),
            "{ctx}: lookup client calls not made: {left:?}"
        );
    }
    std::mem::take(&mut *build_times.lock().unwrap())
}

/// Per-step `build_connector_meta` cost on a trace (`MOONCAKE_BENCH_TRACE`),
/// replayed and checked like the others. Run with `--release --ignored`.
#[test]
#[ignore]
fn bench_mooncake_store_build_connector_meta() {
    let path =
        PathBuf::from(std::env::var_os("MOONCAKE_BENCH_TRACE").expect("MOONCAKE_BENCH_TRACE"));
    let mut times: Vec<f64> = replay(&path).iter().map(|d| d.as_secs_f64() * 1e6).collect();
    let total: f64 = times.iter().sum();
    let n = times.len();
    times.sort_by(f64::total_cmp);
    println!(
        "build_connector_meta over {n} steps: mean {:.2} us, p50 {:.2} us, p99 {:.2} us, max {:.2} us",
        total / n as f64,
        times[n / 2],
        times[n * 99 / 100],
        times[n - 1],
    );
}

#[test]
fn matches_python_mooncake_store_traces() {
    replay_all(
        "mooncake",
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tests/mooncake_traces"),
        "jsonl",
        |path| {
            replay(path);
        },
    );
}
