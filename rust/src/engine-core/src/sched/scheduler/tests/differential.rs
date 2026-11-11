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
use crate::kv_cache::{BoundaryStateOffload, KvCacheBlocks};
use crate::sched::output::Opaque;
use crate::sched::request::{NewRequest, SamplingParams};
use crate::test_util::sched_trace::{
    eco_json, first_diff, floats_match, so_json, trace_setup, usize_of,
};
use crate::test_util::trace_replay::replay_all;

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

struct Replay {
    scheduler: Scheduler,
    hasher: Arc<dyn crate::sched::request::BlockHasher>,
    matches: MatchAnswers,
    log: CallLog,
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
    replay_all(
        "sched",
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/sched/scheduler/tests/traces"),
        "jsonl",
        replay,
    );
}
