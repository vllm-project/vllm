// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Ports of scenarios from `tests/v1/core/test_scheduler.py`,
//! `test_async_scheduler.py`, `test_prefix_replay.py` and
//! `test_priority_scheduler_random.py`.

mod differential;
mod mooncake_store;
mod nixl_differential;

use std::sync::Arc;

use super::*;
use crate::hash::{BlockHash, HashAlgo, RequestBlockHasher, RequestExtraKeys};
use crate::kv_cache::{KvCacheBlocks, KvCacheGroupSpec, KvCacheSpec, KvCacheSpecType};
use crate::sched::multimodal::{MultiModalData, MultiModalFeatureSpec, PlaceholderRange};
use crate::sched::output::{CachedRequestData, Opaque};
use crate::sched::request::{BlockHasher, FinishReason, NewRequest, SamplingParams, StopReason};

const EOS: i32 = 50256;

struct TestHasher(RequestBlockHasher);

impl BlockHasher for TestHasher {
    fn hash_new_blocks(&self, request: &Request) -> Vec<BlockHash> {
        self.0.new_block_hashes(
            request.all_token_ids(),
            request.block_hashes.len(),
            request.block_hashes.last(),
            &RequestExtraKeys::default(),
        )
    }
}

fn hasher(block_size: usize) -> Arc<dyn BlockHasher> {
    Arc::new(TestHasher(RequestBlockHasher::new(
        HashAlgo::Sha256,
        BlockHash::from_slice(&[0; 32]),
        block_size,
        0,
    )))
}

fn full_attention(block_size: usize) -> KvCacheGroupSpec {
    KvCacheGroupSpec::new(KvCacheSpec::new(KvCacheSpecType::FullAttention, block_size))
}

/// `create_scheduler()`: a single full-attention group of `num_blocks`.
fn create_scheduler(
    num_blocks: usize,
    block_size: usize,
    configure: impl FnOnce(&mut SchedulerConfig),
) -> Scheduler {
    create_scheduler_with_groups(
        num_blocks,
        vec![full_attention(block_size)],
        configure,
        None,
    )
}

fn create_scheduler_with_groups(
    num_blocks: usize,
    groups: Vec<KvCacheGroupSpec>,
    configure: impl FnOnce(&mut SchedulerConfig),
    connector: Option<Box<dyn KvConnector>>,
) -> Scheduler {
    let block_size = groups[0].kv_cache_spec.block_size;
    let mut config = SchedulerConfig::for_test(block_size, 8192);
    config.num_gpu_blocks = num_blocks;
    config.log_stats = true;
    if connector.is_some() {
        config.kv_transfer = Some(KvTransferConfig {
            is_kv_producer: true,
            is_kv_consumer: true,
            recompute_kv_load_failures: true,
        });
    }
    configure(&mut config);
    let mut scheduler = Scheduler::new(
        config,
        KvCacheConfig::new(num_blocks, groups),
        connector,
        Box::new(AcceptAllStructuredOutput),
    )
    .unwrap();
    scheduler.set_clock(Box::new(|| 0.0));
    scheduler
}

struct RequestSpec {
    num_tokens: usize,
    max_tokens: usize,
    ignore_eos: bool,
    stop_token_ids: Vec<i32>,
    same_prompt: bool,
    block_size: usize,
}

impl Default for RequestSpec {
    fn default() -> Self {
        Self {
            num_tokens: 10,
            max_tokens: 16,
            ignore_eos: false,
            stop_token_ids: Vec::new(),
            same_prompt: false,
            block_size: 16,
        }
    }
}

/// `create_requests()`: request `i` has id `"{i}"` and prompt `[i] * num_tokens`.
fn create_requests(num_requests: usize, spec: RequestSpec) -> Vec<Request> {
    (0..num_requests)
        .map(|i| {
            let token = if spec.same_prompt { 0 } else { i as i32 };
            Request::new(NewRequest {
                sampling_params: Some(SamplingParams {
                    max_tokens: Some(spec.max_tokens),
                    eos_token_id: (!spec.ignore_eos).then_some(EOS),
                    stop_token_ids: spec.stop_token_ids.clone(),
                    ..Default::default()
                }),
                block_hasher: Some(hasher(spec.block_size)),
                ..NewRequest::for_test(&i.to_string(), vec![token; spec.num_tokens])
            })
            .unwrap()
        })
        .collect()
}

fn model_output(req_ids: &[&str], sampled_token_ids: Vec<Vec<i32>>) -> ModelRunnerOutput {
    ModelRunnerOutput {
        req_ids: req_ids.iter().map(|s| s.to_string()).collect(),
        sampled_token_ids,
        ..Default::default()
    }
}

/// One sampled token `i` per scheduled request, in schedule order.
fn sample_all(output: &SchedulerOutput) -> ModelRunnerOutput {
    ModelRunnerOutput {
        req_ids: output.num_scheduled_tokens.keys().cloned().collect(),
        sampled_token_ids: (0..output.num_scheduled_tokens.len()).map(|i| vec![i as i32]).collect(),
        ..Default::default()
    }
}

impl Scheduler {
    fn req(&self, request_id: &str) -> &Request {
        self.get_request(request_id).unwrap()
    }

    fn running_ids(&self) -> Vec<String> {
        self.running_request_ids().map(str::to_owned).collect()
    }

    /// Put a request straight into the running queue, as Python tests do by
    /// appending to `scheduler.running`; pair with `update_after_schedule` on
    /// a hand-made output.
    fn force_running(&mut self, mut request: Request) {
        request.num_computed_tokens = request.num_tokens();
        request.status = RequestStatus::Running;
        let h = self.requests.insert(request);
        self.running.push(h);
    }
}

/// Requests in the order they were scheduled, with token counts.
fn scheduled(output: &SchedulerOutput) -> Vec<(String, usize)> {
    output.num_scheduled_tokens.iter().map(|(k, &v)| (k.clone(), v)).collect()
}

#[test]
fn schedule_admits_all_requests() {
    for enable_prefix_caching in [false, true] {
        let mut scheduler = create_scheduler(10000, 16, |c| {
            c.enable_prefix_caching = enable_prefix_caching;
        });
        for request in create_requests(10, RequestSpec::default()) {
            scheduler.add_request(request);
        }

        let output = scheduler.schedule(false);
        assert_eq!(output.scheduled_new_reqs.len(), 10);
        assert_eq!(output.scheduled_cached_reqs.num_reqs(), 0);
        assert!(output.finished_req_ids.is_empty());
        for (req_id, num_tokens) in &output.num_scheduled_tokens {
            assert_eq!(*num_tokens, scheduler.req(req_id).num_prompt_tokens);
        }
        assert!(scheduler.waiting.is_empty());
        let expected: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        assert_eq!(scheduler.running_ids(), expected);
    }
}

#[test]
fn scheduler_output_snapshot() {
    let mut scheduler = create_scheduler(100, 16, |c| c.enable_prefix_caching = true);
    for request in create_requests(
        2,
        RequestSpec {
            num_tokens: 20,
            ..Default::default()
        },
    ) {
        scheduler.add_request(request);
    }
    let output = scheduler.schedule(false);
    scheduler.update_from_output(&output, &sample_all(&output));
    let output = scheduler.schedule(false);
    expect_test::expect![[r#"
        SchedulerOutput {
            scheduled_new_reqs: [],
            scheduled_cached_reqs: CachedRequestData {
                req_ids: [
                    "0",
                    "1",
                ],
                new_token_ids: [],
                new_block_ids: [
                    None,
                    None,
                ],
                num_computed_tokens: [
                    20,
                    20,
                ],
                num_output_tokens: [
                    1,
                    1,
                ],
                req_id_to_num_output_tokens: OnceLock(
                    <uninit>,
                ),
            },
            num_scheduled_tokens: {
                "0": 1,
                "1": 1,
            },
            total_num_scheduled_tokens: 2,
            scheduled_spec_decode_tokens: {},
            scheduled_encoder_inputs: {},
            finished_req_ids: {},
            free_encoder_mm_hashes: [],
            encoder_input_updates: {},
            iteration_details: None,
            preempted_req_ids: Some(
                {},
            ),
            has_structured_output_requests: false,
            has_structured_output_spec_decode: false,
            pending_structured_output_tokens: false,
            num_invalid_spec_tokens: None,
            has_sync_kv_loads: false,
            new_block_ids_to_zero: None,
            kv_cache_block_copies: None,
            num_spec_tokens_to_schedule: 0,
            skip_drafts: false,
            kv_connector_metadata: None,
            aux_output_connector_metadata: None,
            ec_connector_metadata: None,
        }
    "#]]
    .assert_debug_eq(&output);
}

#[test]
fn schedule_order_is_fcfs() {
    for enable_chunked_prefill in [false, true] {
        let mut scheduler = create_scheduler(10000, 16, |c| {
            c.enable_chunked_prefill = enable_chunked_prefill;
            c.max_num_batched_tokens = 1024;
        });
        // long requests
        for (i, mut r) in create_requests(
            2,
            RequestSpec {
                num_tokens: 800,
                ..Default::default()
            },
        )
        .into_iter()
        .enumerate()
        {
            r.request_id = format!("long{i}");
            scheduler.add_request(r);
        }
        // short requests
        for r in create_requests(
            2,
            RequestSpec {
                num_tokens: 10,
                ..Default::default()
            },
        ) {
            scheduler.add_request(r);
        }
        let output = scheduler.schedule(false);
        if enable_chunked_prefill {
            // When enable_chunked_prefill is True, the second long request is
            // chunked and the short requests are not scheduled.
            assert_eq!(
                scheduled(&output),
                [("long0".to_owned(), 800), ("long1".to_owned(), 224)]
            );
        } else {
            // When enable_chunked_prefill is False, scheduling stops at the
            // request that does not fit.
            assert_eq!(scheduled(&output), [("long0".to_owned(), 800)]);
        }
    }
}

#[test]
fn stop_via_update_from_output() {
    // Test case 1: Stop on EOS token
    let mut scheduler = create_scheduler(10000, 16, |c| c.num_spec_tokens = 1);
    for req in create_requests(
        2,
        RequestSpec {
            max_tokens: 10,
            ..Default::default()
        },
    ) {
        scheduler.force_running(req);
    }
    let mut scheduler_output = SchedulerOutput {
        num_scheduled_tokens: [("0".to_owned(), 1), ("1".to_owned(), 2)].into_iter().collect(),
        total_num_scheduled_tokens: 3,
        scheduled_spec_decode_tokens: [("0".to_owned(), vec![]), ("1".to_owned(), vec![10])]
            .into_iter()
            .collect(),
        scheduled_cached_reqs: CachedRequestData::make_empty(),
        ..Default::default()
    };
    scheduler.update_after_schedule(&mut scheduler_output);
    // First request hits EOS, second continues
    let out = scheduler.update_from_output(
        &scheduler_output,
        &model_output(&["0", "1"], vec![vec![EOS], vec![10, 11]]),
    );
    assert_eq!(scheduler.running_ids(), ["1"]);
    assert!(scheduler.finished_req_ids.contains("0"));
    assert_eq!(scheduler.req("1").output_token_ids(), [10, 11]);
    let outputs = &out.engine_core_outputs[&0].outputs;
    assert_eq!(outputs[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(outputs[0].new_token_ids, [EOS]);
    // The finished request is gone.
    assert!(!scheduler.has_request("0"));
    let removed: Vec<String> =
        scheduler.take_removed_requests().into_iter().map(|r| r.request_id).collect();
    assert_eq!(removed, ["0"]);

    // Test case 2: Stop on custom stop token
    let mut scheduler = create_scheduler(10000, 16, |c| c.num_spec_tokens = 2);
    for req in create_requests(
        2,
        RequestSpec {
            max_tokens: 10,
            stop_token_ids: vec![42, 43],
            ..Default::default()
        },
    ) {
        scheduler.force_running(req);
    }
    let mut scheduler_output = SchedulerOutput {
        num_scheduled_tokens: [("0".to_owned(), 3), ("1".to_owned(), 2)].into_iter().collect(),
        total_num_scheduled_tokens: 5,
        scheduled_spec_decode_tokens: [("0".to_owned(), vec![10, 42]), ("1".to_owned(), vec![13])]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    scheduler.update_after_schedule(&mut scheduler_output);
    let out = scheduler.update_from_output(
        &scheduler_output,
        &model_output(&["0", "1"], vec![vec![10, 42, 12], vec![13, 14]]),
    );
    assert_eq!(scheduler.running_ids(), ["1"]);
    let stopped = &out.engine_core_outputs[&0].outputs[0];
    assert_eq!(stopped.stop_reason, Some(StopReason::TokenId(42)));
    assert_eq!(stopped.new_token_ids, [10, 42]);
    assert_eq!(scheduler.req("1").output_token_ids(), [13, 14]);

    // Test case 3: Stop on max tokens
    let mut scheduler = create_scheduler(10000, 16, |c| c.num_spec_tokens = 2);
    for req in create_requests(
        2,
        RequestSpec {
            max_tokens: 2,
            ..Default::default()
        },
    ) {
        scheduler.force_running(req);
    }
    let mut scheduler_output = SchedulerOutput {
        num_scheduled_tokens: [("0".to_owned(), 3), ("1".to_owned(), 1)].into_iter().collect(),
        total_num_scheduled_tokens: 4,
        scheduled_spec_decode_tokens: [("0".to_owned(), vec![10, 11]), ("1".to_owned(), vec![])]
            .into_iter()
            .collect(),
        ..Default::default()
    };
    scheduler.update_after_schedule(&mut scheduler_output);
    let out = scheduler.update_from_output(
        &scheduler_output,
        &model_output(&["0", "1"], vec![vec![10, 11, 12], vec![13]]),
    );
    assert_eq!(scheduler.running_ids(), ["1"]);
    let stopped = &out.engine_core_outputs[&0].outputs[0];
    assert_eq!(stopped.finish_reason, Some(FinishReason::Length));
    // Truncated to max_tokens
    assert_eq!(stopped.new_token_ids, [10, 11]);
    assert_eq!(scheduler.req("1").output_token_ids(), [13]);

    // Test case 4: Ignore EOS flag
    let mut scheduler = create_scheduler(10000, 16, |c| c.num_spec_tokens = 2);
    for req in create_requests(
        1,
        RequestSpec {
            max_tokens: 10,
            ignore_eos: true,
            ..Default::default()
        },
    ) {
        scheduler.force_running(req);
    }
    let mut scheduler_output = SchedulerOutput {
        num_scheduled_tokens: [("0".to_owned(), 3)].into_iter().collect(),
        total_num_scheduled_tokens: 3,
        scheduled_spec_decode_tokens: [("0".to_owned(), vec![EOS, 10])].into_iter().collect(),
        ..Default::default()
    };
    scheduler.update_after_schedule(&mut scheduler_output);
    scheduler.update_from_output(
        &scheduler_output,
        &model_output(&["0"], vec![vec![EOS, 10, 11]]),
    );
    assert_eq!(scheduler.running_ids(), ["0"]);
    assert!(!scheduler.req("0").is_finished());
    assert_eq!(scheduler.req("0").output_token_ids(), [EOS, 10, 11]);
}

#[test]
fn preempt_during_execution() {
    // NOTE(woosuk): The actual number of available blocks is 10 instead of 11
    // because block 0 is reserved as the null block.
    let mut scheduler = create_scheduler(11, 16, |c| c.max_num_batched_tokens = 100);
    let mut requests = create_requests(
        2,
        RequestSpec {
            num_tokens: 80,
            ..Default::default()
        },
    )
    .into_iter();

    // Schedule the first request.
    scheduler.add_request(requests.next().unwrap());
    let scheduler_output0 = scheduler.schedule(false);
    assert_eq!(scheduler_output0.num_scheduled_tokens.len(), 1);
    assert_eq!(
        scheduler_output0.scheduled_new_reqs[0].block_ids[0].len(),
        5
    );

    // Schedule the second request while the first request is still running.
    // This scenario can occur in certain cases, when max_concurrent_batches > 1
    // (e.g., when pipeline parallelism is used).
    scheduler.add_request(requests.next().unwrap());
    let scheduler_output1 = scheduler.schedule(false);
    assert_eq!(scheduler_output1.num_scheduled_tokens.len(), 1);
    assert_eq!(
        scheduler_output1.scheduled_new_reqs[0].block_ids[0].len(),
        5
    );

    // Get the output of the first request.
    scheduler.update_from_output(&scheduler_output0, &model_output(&["0"], vec![vec![0]]));

    // Schedule the first request again. This will cause the preemption
    // of the second request because the KV cache is full.
    let output = scheduler.schedule(false);
    assert_eq!(scheduler.running_ids(), ["0"]);
    assert_eq!(scheduler.req("1").status, RequestStatus::Preempted);
    assert_eq!(output.preempted_req_ids.as_ref().unwrap().len(), 1);

    scheduler.update_from_output(&scheduler_output1, &model_output(&["1"], vec![vec![42]]));

    // The second request (that is preempted) should be updated with the
    // sampled token id.
    assert_eq!(scheduler.req("1").output_token_ids(), [42]);
}

fn priority_request(id: &str, priority: i64, arrival_time: f64, num_tokens: usize) -> Request {
    Request::new(NewRequest {
        priority,
        arrival_time: Some(arrival_time),
        sampling_params: Some(SamplingParams {
            eos_token_id: Some(EOS),
            ..Default::default()
        }),
        block_hasher: Some(hasher(16)),
        ..NewRequest::for_test(id, vec![0; num_tokens])
    })
    .unwrap()
}

/// Under KV block pressure the scheduler preempts the lowest-priority
/// *running* request, not the highest-priority one.
#[test]
fn priority_scheduling_preemption() {
    let mut scheduler = create_scheduler(6, 16, |c| {
        c.policy = SchedulingPolicy::Priority;
        c.max_num_seqs = 3;
        c.max_num_batched_tokens = 200;
    });

    // --- Phase 1: low-priority request starts running ---
    scheduler.add_request(priority_request("lo1", 5, 1.0, 32));
    let output = scheduler.schedule(false);
    assert_eq!(output.scheduled_new_reqs.len(), 1);
    // Decode: lo1 now has 33 tokens (crosses 32-token boundary).
    scheduler.update_from_output(&output, &model_output(&["lo1"], vec![vec![100]]));

    // --- Phase 2: high-priority request arrives AFTER lo1 is running ---
    scheduler.add_request(priority_request("hi1", 0, 2.0, 32));
    let output = scheduler.schedule(false);
    assert!(output.scheduled_new_reqs.iter().any(|r| r.req_id == "hi1"));
    assert_eq!(scheduler.running.len(), 2);
    scheduler.update_from_output(
        &output,
        &model_output(&["lo1", "hi1"], vec![vec![101], vec![100]]),
    );

    // --- Phase 3: preemption with mixed-priority running requests ---
    let output = scheduler.schedule(false);
    assert_eq!(scheduler.req("lo1").status, RequestStatus::Preempted);
    assert_eq!(scheduler.running_ids(), ["hi1"]);
    assert_eq!(scheduled(&output), [("hi1".to_owned(), 1)]);
}

#[test]
fn priority_scheduling_orders_waiting_requests() {
    let mut scheduler = create_scheduler(10000, 16, |c| {
        c.policy = SchedulingPolicy::Priority;
        c.max_num_seqs = 2;
    });
    // Lower priority value goes first; equal priority by arrival time.
    for (id, priority, arrival) in [("a", 2, 0.0), ("b", 1, 3.0), ("c", 1, 2.0), ("d", 0, 9.0)] {
        scheduler.add_request(priority_request(id, priority, arrival, 10));
    }
    let output = scheduler.schedule(false);
    assert_eq!(
        scheduled(&output),
        [("d".to_owned(), 10), ("c".to_owned(), 10)]
    );
}

#[test]
fn prefix_cache_hit_skips_cached_blocks() {
    let mut scheduler = create_scheduler(10000, 16, |c| c.enable_prefix_caching = true);
    let mut requests = create_requests(
        2,
        RequestSpec {
            num_tokens: 50,
            same_prompt: true,
            ..Default::default()
        },
    )
    .into_iter();
    scheduler.add_request(requests.next().unwrap());
    let output = scheduler.schedule(false);
    scheduler.update_from_output(&output, &sample_all(&output));

    scheduler.add_request(requests.next().unwrap());
    let output = scheduler.schedule(false);
    // 3 full blocks (48 tokens) hit.
    let new_req = &output.scheduled_new_reqs[0];
    assert_eq!(new_req.num_computed_tokens, 48);
    assert_eq!(output.num_scheduled_tokens["1"], 2);
    let blocks0 = scheduler.kv_cache_manager.get_block_ids("0");
    assert_eq!(new_req.block_ids[0][..3], blocks0[0][..3]);

    // The first request's lookup was reported with the previous step's
    // stats.
    let stats = scheduler.make_stats(None).unwrap();
    assert_eq!(
        (
            stats.prefix_cache_stats.requests,
            stats.prefix_cache_stats.queries,
            stats.prefix_cache_stats.hits
        ),
        (1, 50, 48)
    );
}

#[test]
fn long_prefill_threshold_applies_with_other_requests() {
    let mut scheduler = create_scheduler(10000, 16, |c| c.long_prefill_token_threshold = 100);
    for r in create_requests(
        2,
        RequestSpec {
            num_tokens: 300,
            ..Default::default()
        },
    ) {
        scheduler.add_request(r);
    }
    let output = scheduler.schedule(false);
    assert_eq!(
        scheduled(&output),
        [("0".to_owned(), 100), ("1".to_owned(), 100)]
    );

    // Alone, a request uses the whole budget.
    let mut scheduler = create_scheduler(10000, 16, |c| c.long_prefill_token_threshold = 100);
    for r in create_requests(
        1,
        RequestSpec {
            num_tokens: 300,
            ..Default::default()
        },
    ) {
        scheduler.add_request(r);
    }
    assert_eq!(scheduler.schedule(false).num_scheduled_tokens["0"], 300);
}

#[test]
fn spec_decode_schedules_drafts_and_rolls_back_rejections() {
    let mut scheduler = create_scheduler(10000, 16, |c| c.num_spec_tokens = 2);
    for r in create_requests(1, RequestSpec::default()) {
        scheduler.add_request(r);
    }
    let output = scheduler.schedule(false);
    scheduler.update_from_output(&output, &model_output(&["0"], vec![vec![1]]));
    scheduler.update_draft_token_ids(DraftTokenIds {
        req_ids: vec!["0".to_owned()],
        draft_token_ids: vec![vec![5, 6]],
    });

    let output = scheduler.schedule(false);
    assert_eq!(output.num_scheduled_tokens["0"], 3);
    assert_eq!(output.scheduled_spec_decode_tokens["0"], [5, 6]);
    assert_eq!(scheduler.req("0").num_computed_tokens, 13);

    // One draft accepted, one rejected.
    let out = scheduler.update_from_output(&output, &model_output(&["0"], vec![vec![5, 7]]));
    assert_eq!(scheduler.req("0").num_computed_tokens, 12);
    assert_eq!(scheduler.req("0").output_token_ids(), [1, 5, 7]);
    let stats = out.engine_core_outputs[&0].scheduler_stats.as_ref().unwrap();
    expect_test::expect![[r#"
        Some(
            SpecDecodingStats {
                num_spec_tokens: 2,
                num_drafts: 1,
                num_draft_tokens: 2,
                num_accepted_tokens: 1,
                num_accepted_tokens_per_pos: [
                    1,
                    0,
                ],
                num_draft_tokens_per_pos: [
                    1,
                    1,
                ],
            },
        )
    "#]]
    .assert_debug_eq(&stats.spec_decoding_stats);
}

/// A new decode request is padded to the uniform spec decode size.
#[test]
fn spec_decode_padding_first_decode_step() {
    let mut scheduler = create_scheduler(10000, 16, |c| c.num_spec_tokens = 2);
    let mut requests = create_requests(2, RequestSpec::default()).into_iter();
    scheduler.add_request(requests.next().unwrap());
    let output = scheduler.schedule(false);
    scheduler.update_from_output(&output, &model_output(&["0"], vec![vec![1]]));
    // A resumed request with one token left to compute and a running request
    // in the batch.
    let mut resumed = requests.next().unwrap();
    resumed.append_output_token_ids(&[3]);
    scheduler.add_request(resumed);
    let h = scheduler.requests.handle("1").unwrap();
    scheduler.requests[h].num_computed_tokens = 0;
    let output = scheduler.schedule(false);
    // "0" decodes one token (no drafts); "1" prefills all 11 tokens.
    assert_eq!(
        scheduled(&output),
        [("0".to_owned(), 1), ("1".to_owned(), 11)]
    );
}

#[test]
fn async_scheduling_stops_by_max_tokens() {
    for max_tokens in [1, 2, 3, 5] {
        let mut scheduler = create_scheduler(10000, 16, |c| {
            c.async_scheduling = true;
            c.scheduler_class = SchedulerClass::Async { decode_stagger: 1 };
        });
        let mut requests = create_requests(
            2,
            RequestSpec {
                max_tokens,
                ..Default::default()
            },
        )
        .into_iter();
        let (req0, req1) = (requests.next().unwrap(), requests.next().unwrap());
        let mut expected_total = req0.num_prompt_tokens + max_tokens - 1;
        expected_total += req1.num_prompt_tokens + max_tokens - 1;

        let mut sched_outputs = std::collections::VecDeque::new();
        scheduler.add_request(req0);
        sched_outputs.push_back(scheduler.schedule(false));
        scheduler.add_request(req1);
        sched_outputs.push_back(scheduler.schedule(false));

        let mut total = 0;
        let mut num_output_tokens = std::collections::HashMap::new();
        while let Some(sched_output) = sched_outputs.pop_front() {
            total += sched_output.total_num_scheduled_tokens;
            let out = scheduler.update_from_output(&sched_output, &sample_all(&sched_output));
            for eco in out.engine_core_outputs.values() {
                for o in &eco.outputs {
                    *num_output_tokens.entry(o.request_id.clone()).or_insert(0) +=
                        o.new_token_ids.len();
                }
            }
            let sched_output = scheduler.schedule(false);
            if !sched_output.num_scheduled_tokens.is_empty() {
                sched_outputs.push_back(sched_output);
            }
        }
        assert_eq!(scheduler.get_num_unfinished_requests(), 0);
        assert_eq!(num_output_tokens["0"], max_tokens);
        assert_eq!(num_output_tokens["1"], max_tokens);
        // Ensure we aren't scheduling more tokens than necessary.
        assert_eq!(total, expected_total, "max_tokens={max_tokens}");
    }
}

#[test]
fn abort_waiting_and_running_requests() {
    let mut scheduler = create_scheduler(10000, 16, |c| c.max_num_seqs = 1);
    for r in create_requests(2, RequestSpec::default()) {
        scheduler.add_request(r);
    }
    let output = scheduler.schedule(false);
    assert_eq!(scheduler.get_request_counts(), (1, 1));
    let aborted =
        scheduler.finish_requests(Some(&["1", "0", "missing"]), RequestStatus::FinishedAborted);
    assert_eq!(aborted, ["1", "0"]);
    assert_eq!(scheduler.get_request_counts(), (0, 0));
    assert!(!scheduler.has_unfinished_requests());
    assert!(scheduler.has_finished_requests());
    assert_eq!(scheduler.kv_cache_manager.usage(), 0.0);

    // The in-flight output of an aborted request is dropped.
    let out = scheduler.update_from_output(&output, &sample_all(&output));
    assert!(out.engine_core_outputs[&0].outputs.is_empty());
    let output = scheduler.schedule(false);
    let mut finished: Vec<_> = output.finished_req_ids.iter().cloned().collect();
    finished.sort();
    assert_eq!(finished, ["0", "1"]);
    assert!(!scheduler.has_finished_requests());
}

#[test]
fn reset_prefix_cache_preempts_running_requests() {
    let mut scheduler = create_scheduler(10000, 16, |c| c.enable_prefix_caching = true);
    for r in create_requests(
        2,
        RequestSpec {
            num_tokens: 32,
            ..Default::default()
        },
    ) {
        scheduler.add_request(r);
    }
    let output = scheduler.schedule(false);
    scheduler.update_from_output(&output, &sample_all(&output));
    assert!(!scheduler.reset_prefix_cache(false, false));
    assert!(scheduler.reset_prefix_cache(true, false));
    assert_eq!(scheduler.get_request_counts(), (0, 2));
    // Preempted in reverse, so they come back in FIFO order.
    let output = scheduler.schedule(false);
    assert_eq!(
        scheduled(&output),
        [("0".to_owned(), 33), ("1".to_owned(), 33)]
    );
}

#[test]
fn schedule_partial_requests_by_encoder_budget() {
    let mut scheduler = create_scheduler(10000, 16, |c| {
        c.max_num_batched_tokens = 1024;
        c.max_num_encoder_input_tokens = 1024;
        c.encoder_cache_size = 1024;
    });
    for mut r in create_requests(
        3,
        RequestSpec {
            num_tokens: 800,
            ..Default::default()
        },
    ) {
        r.mm_features = vec![MultiModalFeatureSpec {
            modality: "image".into(),
            identifier: format!("hash{}_0", r.request_id),
            mm_position: PlaceholderRange::new(100, 600, None),
            mm_hash: None,
            data: MultiModalData::Payload,
        }];
        scheduler.add_request(r);
    }

    let output = scheduler.schedule(false);
    // The first request is scheduled fully; the others stop before their
    // <img> tokens because of the encoder budget.
    assert_eq!(
        scheduled(&output),
        [
            ("0".to_owned(), 800),
            ("1".to_owned(), 100),
            ("2".to_owned(), 100)
        ]
    );
    assert_eq!(output.scheduled_encoder_inputs["0"], [0]);
    scheduler.update_from_output(
        &output,
        &model_output(&["0", "1", "2"], vec![vec![0], vec![], vec![]]),
    );

    // Only the first and second requests are scheduled.
    let output = scheduler.schedule(false);
    assert_eq!(scheduler.running.len(), 3);
    assert_eq!(
        scheduled(&output),
        [("0".to_owned(), 1), ("1".to_owned(), 700)]
    );
    // The first request's consumed image payload is released.
    expect_test::expect![[r#"
        {
            "0": {
                0: Release,
            },
        }
    "#]]
    .assert_debug_eq(&output.encoder_input_updates);
}

// ---- SWA bounded replay (`test_prefix_replay.py`) ----

const WINDOW: usize = 32;
const NUM_PROMPT_TOKENS: usize = 100;
// 100 tokens -> 6 full blocks cached -> 96-token hit.
const HIT_TOKENS: usize = NUM_PROMPT_TOKENS / 16 * 16;
const FULL: usize = 0;
const SWA: usize = 1;

fn replay_groups() -> Vec<KvCacheGroupSpec> {
    vec![
        full_attention(16),
        KvCacheGroupSpec::new(KvCacheSpec::new(
            KvCacheSpecType::SlidingWindowMla {
                sliding_window: WINDOW,
                extra_retained_tokens: 0,
                bounded_replay: true,
            },
            16,
        )),
    ]
}

fn replay_scheduler(
    long_prefill_token_threshold: usize,
    connector: Option<Box<dyn KvConnector>>,
) -> Scheduler {
    let scheduler = create_scheduler_with_groups(
        10000,
        replay_groups(),
        |c| {
            c.enable_prefix_caching = true;
            c.long_prefill_token_threshold = long_prefill_token_threshold;
        },
        connector,
    );
    assert_eq!(scheduler.prefix_replay_tokens, WINDOW);
    scheduler
}

/// Model output for every scheduled request: a token once its prefill is
/// done, nothing while it is still prefilling.
fn step_output(scheduler: &Scheduler, out: &SchedulerOutput) -> ModelRunnerOutput {
    let req_ids: Vec<&str> = out.num_scheduled_tokens.keys().map(String::as_str).collect();
    let sampled = req_ids
        .iter()
        .map(|id| {
            let r = scheduler.req(id);
            if r.num_computed_tokens >= r.num_prompt_tokens {
                vec![1000]
            } else {
                vec![]
            }
        })
        .collect();
    model_output(&req_ids, sampled)
}

fn replay_requests() -> (Request, Request) {
    let mut requests = create_requests(
        2,
        RequestSpec {
            num_tokens: NUM_PROMPT_TOKENS,
            same_prompt: true,
            ..Default::default()
        },
    )
    .into_iter();
    (requests.next().unwrap(), requests.next().unwrap())
}

#[test]
fn hit_replays_window_without_reallocating() {
    let mut scheduler = replay_scheduler(0, None);
    let (first, second) = replay_requests();
    scheduler.add_request(first);
    let out = scheduler.schedule(false);
    assert_eq!(out.num_scheduled_tokens["0"], NUM_PROMPT_TOKENS);
    let step = step_output(&scheduler, &out);
    scheduler.update_from_output(&out, &step);

    scheduler.add_request(second);
    let out = scheduler.schedule(false);
    let replay_start = HIT_TOKENS - WINDOW;
    let new_req = &out.scheduled_new_reqs[0];
    assert_eq!(new_req.num_computed_tokens, replay_start);
    assert_eq!(new_req.replay_start, replay_start);
    assert_eq!(
        out.num_scheduled_tokens["1"],
        NUM_PROMPT_TOKENS - replay_start
    );

    let first_blocks = scheduler.kv_cache_manager.get_block_ids("0");
    let second_blocks = scheduler.kv_cache_manager.get_block_ids("1");
    // The full-attention hit is shared (replayed tokens included); only the
    // tail block is new.
    let num_hit_blocks = HIT_TOKENS / 16;
    assert_eq!(
        second_blocks[FULL][..num_hit_blocks],
        first_blocks[FULL][..num_hit_blocks]
    );
    assert_eq!(second_blocks[FULL].len(), num_hit_blocks + 1);
    // The window group never hits: blocks below the window are null.
    let num_skipped_blocks = (HIT_TOKENS - WINDOW + 1) / 16;
    assert!(second_blocks[SWA][..num_skipped_blocks].iter().all(|&b| b == 0));
    assert!(second_blocks[SWA][num_skipped_blocks..].iter().all(|&b| b != 0));

    let step = step_output(&scheduler, &out);
    scheduler.update_from_output(&out, &step);
    assert_eq!(scheduler.req("1").num_computed_tokens, NUM_PROMPT_TOKENS);
}

#[test]
fn replay_only_chunks_make_progress() {
    let mut scheduler = replay_scheduler(16, None);
    let (first, second) = replay_requests();
    scheduler.add_request(first);
    while scheduler.req("0").num_computed_tokens < NUM_PROMPT_TOKENS {
        let out = scheduler.schedule(false);
        let step = step_output(&scheduler, &out);
        scheduler.update_from_output(&out, &step);
    }

    scheduler.add_request(second);
    let mut chunks = Vec::new();
    while scheduler.req("1").num_computed_tokens < NUM_PROMPT_TOKENS {
        let out = scheduler.schedule(false);
        if chunks.is_empty() {
            let new_req = out.scheduled_new_reqs.iter().find(|r| r.req_id == "1").unwrap();
            assert_eq!(new_req.num_computed_tokens, HIT_TOKENS - WINDOW);
            // A replay-only chunk adopts the hit and allocates nothing new.
            assert_eq!(
                scheduler.kv_cache_manager.get_block_ids("1")[FULL].len(),
                HIT_TOKENS / 16
            );
        }
        chunks.push(out.num_scheduled_tokens["1"]);
        let step = step_output(&scheduler, &out);
        scheduler.update_from_output(&out, &step);
    }
    // 32 replayed + 4 new tokens in 16-token chunks: two replay-only chunks.
    assert_eq!(chunks, [16, 16, NUM_PROMPT_TOKENS - HIT_TOKENS]);
    assert_eq!(scheduler.req("1").status, RequestStatus::Running);
}

// ---- KV connector ----

/// `MockKVConnector` of `tests/v1/kv_connector/unit/utils.py`.
#[derive(Default)]
struct MockConnector {
    matched_tokens: usize,
    is_async: bool,
    /// `get_loaded_kv_cache_group_ids`; the base default is the
    /// prefix-cacheable groups.
    loaded_group_ids: Vec<usize>,
    log: Arc<std::sync::Mutex<Vec<String>>>,
}

impl KvConnector for MockConnector {
    fn supports_divergent_local_hybrid_hits(&self) -> bool {
        false
    }

    fn requires_kv_delivery(&self) -> bool {
        false
    }

    fn on_new_request(&mut self, _request: &Request, _kv_cache_manager: &mut KvCacheManager) {}

    fn may_load_async(&mut self, _request: &Request) -> bool {
        self.is_async
    }

    fn get_num_new_matched_tokens(
        &mut self,
        _request: &mut Request,
        _num_computed_tokens: usize,
        _kv_cache_manager: &mut KvCacheManager,
    ) -> (Option<usize>, bool) {
        (Some(self.matched_tokens), self.is_async)
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
        self.log.lock().unwrap().push(format!(
            "alloc {} groups={} ext={num_external_tokens}",
            request.request_id,
            blocks.blocks.len()
        ));
    }

    fn get_external_cache_hit_sources(
        &mut self,
        _request: &Request,
        num_external_tokens: usize,
    ) -> Option<Vec<(String, usize)>> {
        Some(vec![("disk".to_owned(), num_external_tokens)])
    }

    fn has_pending_block_frees(&mut self, _kv_cache_manager: &mut KvCacheManager) -> bool {
        false
    }

    fn build_connector_meta(
        &mut self,
        scheduler_output: &SchedulerOutput,
        _block_state: &mut KvConnectorBlockState<'_>,
    ) -> Opaque {
        let new: Vec<String> =
            scheduler_output.scheduled_new_reqs.iter().map(|r| r.req_id.clone()).collect();
        Opaque::new(new)
    }

    fn register_finished_partial_tail(
        &mut self,
        _request: &Request,
        _block_ids: &[Vec<BlockId>],
        _partial_tail_offloads: &[crate::kv_cache::BoundaryStateOffload],
        _kv_cache_manager: &mut KvCacheManager,
    ) -> bool {
        false
    }

    fn request_finished(
        &mut self,
        request: &Request,
        _block_ids: &[Vec<BlockId>],
        _kv_cache_manager: &mut KvCacheManager,
    ) -> RequestFinished {
        self.log.lock().unwrap().push(format!("finished {}", request.request_id));
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

fn kv_output(finished_recving: &[&str]) -> ModelRunnerOutput {
    ModelRunnerOutput {
        kv_connector_output: Some(KvConnectorOutput {
            finished_recving: Some(finished_recving.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn kv_connector_async_load() {
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let connector = MockConnector {
        matched_tokens: 64,
        is_async: true,
        loaded_group_ids: vec![0],
        log: log.clone(),
    };
    let mut scheduler = create_scheduler_with_groups(
        10000,
        vec![full_attention(16)],
        |c| c.enable_prefix_caching = true,
        Some(Box::new(connector)),
    );
    for r in create_requests(
        1,
        RequestSpec {
            num_tokens: 100,
            ..Default::default()
        },
    ) {
        scheduler.add_request(r);
    }
    let out = scheduler.schedule(false);
    assert!(out.num_scheduled_tokens.is_empty());
    assert_eq!(
        scheduler.req("0").status,
        RequestStatus::WaitingForRemoteKvs
    );
    assert_eq!(scheduler.kv_fetch_counts.in_progress, 1);
    // The load holds blocks for the external tokens only.
    assert_eq!(scheduler.kv_cache_manager.get_block_ids("0")[0].len(), 4);
    let update = scheduler.update_from_output(&out, &kv_output(&["0"]));
    assert_eq!(scheduler.kv_fetch_counts.completed_waiting, 1);
    let stats = update.engine_core_outputs[&0].scheduler_stats.clone().unwrap();
    let connector_stats = stats.connector_prefix_cache_stats.unwrap();
    assert_eq!(
        (connector_stats.stats.queries, connector_stats.stats.hits),
        (100, 64)
    );
    assert_eq!(connector_stats.hits_by_source["disk"], 64);
    assert_eq!(stats.num_kv_fetch_reqs_by_stage.completed_waiting, 1);

    let out = scheduler.schedule(false);
    assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 64);
    assert_eq!(out.num_scheduled_tokens["0"], 36);
    assert!(!out.has_sync_kv_loads);
    assert_eq!(scheduler.kv_fetch_counts, KvFetchCounts::default());
    let meta = out.kv_connector_metadata.as_ref().unwrap();
    assert_eq!(*meta.0.downcast_ref::<Vec<String>>().unwrap(), ["0"]);

    let out_update = scheduler.update_from_output(&out, &step_output(&scheduler, &out));
    assert_eq!(
        out_update.engine_core_outputs[&0].outputs[0].new_token_ids,
        [1000]
    );
    scheduler.finish_requests(Some(&["0"]), RequestStatus::FinishedAborted);
    expect_test::expect![[r#"
        [
            "alloc 0 groups=1 ext=64",
            "alloc 0 groups=1 ext=0",
            "finished 0",
        ]
    "#]]
    .assert_debug_eq(&log.lock().unwrap());
}

#[test]
fn kv_connector_sync_load() {
    let connector = MockConnector {
        matched_tokens: 48,
        is_async: false,
        loaded_group_ids: vec![0],
        ..Default::default()
    };
    let mut scheduler = create_scheduler_with_groups(
        10000,
        vec![full_attention(16)],
        |_| {},
        Some(Box::new(connector)),
    );
    for r in create_requests(
        1,
        RequestSpec {
            num_tokens: 100,
            ..Default::default()
        },
    ) {
        scheduler.add_request(r);
    }
    let out = scheduler.schedule(false);
    assert!(out.has_sync_kv_loads);
    assert_eq!(out.scheduled_new_reqs[0].num_computed_tokens, 48);
    assert_eq!(out.num_scheduled_tokens["0"], 52);
}

#[test]
fn async_remote_kv_hit_replays_after_load() {
    let matched = 64;
    let connector = MockConnector {
        matched_tokens: matched,
        is_async: true,
        // The prefix-cacheable groups: the window group is not restored.
        loaded_group_ids: vec![FULL],
        ..Default::default()
    };
    let mut scheduler = replay_scheduler(0, Some(Box::new(connector)));
    for r in create_requests(
        1,
        RequestSpec {
            num_tokens: NUM_PROMPT_TOKENS,
            ..Default::default()
        },
    ) {
        scheduler.add_request(r);
    }
    let out = scheduler.schedule(false);
    assert_eq!(
        scheduler.req("0").status,
        RequestStatus::WaitingForRemoteKvs
    );
    scheduler.update_from_output(&out, &kv_output(&["0"]));

    let out = scheduler.schedule(false);
    let new_req = &out.scheduled_new_reqs[0];
    assert_eq!(new_req.num_computed_tokens, matched - WINDOW);
    assert_eq!(new_req.replay_start, matched - WINDOW);
    assert_eq!(
        out.num_scheduled_tokens["0"],
        NUM_PROMPT_TOKENS - matched + WINDOW
    );
}

#[test]
fn kv_load_carrying_window_does_not_replay() {
    for is_async in [false, true] {
        for matched in [WINDOW, 4 * 16 - 1] {
            let connector = MockConnector {
                matched_tokens: matched,
                is_async,
                loaded_group_ids: vec![FULL, SWA],
                ..Default::default()
            };
            let mut scheduler = replay_scheduler(0, Some(Box::new(connector)));
            for r in create_requests(
                1,
                RequestSpec {
                    num_tokens: NUM_PROMPT_TOKENS,
                    ..Default::default()
                },
            ) {
                scheduler.add_request(r);
            }
            let mut out = scheduler.schedule(false);
            if is_async {
                assert_eq!(
                    scheduler.req("0").status,
                    RequestStatus::WaitingForRemoteKvs
                );
                scheduler.update_from_output(&out, &kv_output(&["0"]));
                out = scheduler.schedule(false);
            }
            let new_req = &out.scheduled_new_reqs[0];
            assert_eq!(new_req.num_computed_tokens, matched);
            assert_eq!(new_req.replay_start, 0);
            assert_eq!(out.num_scheduled_tokens["0"], NUM_PROMPT_TOKENS - matched);
        }
    }
}

#[test]
fn unsupported_features_fail_at_construction() {
    let mut config = SchedulerConfig::for_test(16, 1024);
    config.custom_encoder_cache_manager = true;
    let err = Scheduler::new(
        config,
        KvCacheConfig::new(100, vec![full_attention(16)]),
        None,
        Box::new(AcceptAllStructuredOutput),
    )
    .unwrap_err();
    assert!(matches!(err, SchedulerError::Unsupported { .. }));
}

/// Structured output requests wait for their grammar and report compile
/// failures as errors.
#[test]
fn structured_output_grammar_gating() {
    struct Grammars;
    impl StructuredOutput for Grammars {
        fn grammar_status(&mut self, request: &Request) -> GrammarStatus {
            match request.request_id.as_str() {
                "0" => GrammarStatus::Pending,
                "1" => GrammarStatus::Failed {
                    message: "bad".to_owned(),
                },
                _ => GrammarStatus::Ready,
            }
        }
        fn accept_tokens(&mut self, _request: &Request, _new_token_ids: &[i32]) -> bool {
            true
        }
        fn trim_spec_tokens(&mut self, _request: &Request, spec: Vec<i32>) -> Vec<i32> {
            spec
        }
    }
    let mut config = SchedulerConfig::for_test(16, 8192);
    config.log_stats = true;
    let mut scheduler = Scheduler::new(
        config,
        KvCacheConfig::new(1000, vec![full_attention(16)]),
        None,
        Box::new(Grammars),
    )
    .unwrap();
    for mut r in create_requests(3, RequestSpec::default()) {
        r.use_structured_output = true;
        r.status = RequestStatus::WaitingForStructuredOutputGrammar;
        scheduler.add_request(r);
    }
    let out = scheduler.schedule(false);
    assert_eq!(scheduled(&out), [("2".to_owned(), 10)]);
    assert_eq!(scheduler.deferred_waiting.len(), 2);
    let update = scheduler.update_from_output(&out, &model_output(&["2"], vec![vec![1]]));
    let errors: Vec<_> = update.engine_core_outputs[&0]
        .outputs
        .iter()
        .filter_map(|o| o.error.as_ref().map(|e| (o.request_id.clone(), e.message.clone())))
        .collect();
    assert_eq!(
        errors,
        [("1".to_owned(), "Invalid structured outputs: bad".to_owned())]
    );
    let stats = update.engine_core_outputs[&0].scheduler_stats.as_ref().unwrap();
    assert_eq!(
        (stats.num_waiting_reqs, stats.num_skipped_waiting_reqs),
        (0, 1)
    );
}

/// A resumable (streaming-input) request pauses after each chunk and resumes
/// with the next one, dropping the last sampled token.
#[test]
fn streaming_session_resumes_with_next_chunk() {
    let mut scheduler = create_scheduler(10000, 16, |_| {});
    let new = |prompt: Vec<i32>| {
        Request::new(NewRequest {
            resumable: true,
            sampling_params: Some(SamplingParams {
                max_tokens: Some(2),
                ..Default::default()
            }),
            ..NewRequest::for_test("s", prompt)
        })
        .unwrap()
    };
    scheduler.add_request(new(vec![1, 2, 3]));
    let out = scheduler.schedule(false);
    scheduler.update_from_output(&out, &model_output(&["s"], vec![vec![7]]));
    let out = scheduler.schedule(false);
    let update = scheduler.update_from_output(&out, &model_output(&["s"], vec![vec![8]]));
    let output = &update.engine_core_outputs[&0].outputs[0];
    assert_eq!(output.finish_reason, Some(FinishReason::Length));
    assert_eq!(
        scheduler.req("s").status,
        RequestStatus::WaitingForStreamingReq
    );
    assert_eq!(scheduler.get_num_unfinished_requests(), 0);

    // The next chunk: the kept output token 7 joins the prompt.
    scheduler.add_request(new(vec![4, 5]));
    let session = scheduler.req("s");
    assert_eq!(session.status, RequestStatus::Waiting);
    assert_eq!(
        session.prompt_token_ids.as_deref(),
        Some(&[1, 2, 3, 7, 4, 5][..])
    );
    let out = scheduler.schedule(false);
    assert_eq!(out.scheduled_new_reqs.len(), 1);
    assert_eq!(out.num_scheduled_tokens["s"], 2);
}

/// `test_priority_scheduling_blast`: random requests, priorities, preemption
/// and spec decoding; every request eventually finishes and the scheduler's
/// invariants hold at each step.
#[test]
fn priority_scheduling_blast() {
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.next() % (hi - lo + 1)
        }
    }
    for (enable_prefix_caching, num_spec_tokens, num_blocks) in [
        (false, 0, 2000),
        (true, 0, 300),
        (true, 2, 300),
        (false, 2, 100),
    ] {
        let mut rng = Rng(42 + num_blocks as u64);
        let mut scheduler = create_scheduler(num_blocks, 16, |c| {
            c.policy = SchedulingPolicy::Priority;
            c.enable_prefix_caching = enable_prefix_caching;
            c.num_spec_tokens = num_spec_tokens;
            c.max_num_seqs = 32;
        });
        let mut next_id = 0;
        let add = |scheduler: &mut Scheduler, rng: &mut Rng, next_id: &mut usize| {
            let num_tokens = rng.range(1, 400) as usize;
            let prefix = rng.range(0, 3) as i32;
            let r = Request::new(NewRequest {
                priority: rng.range(0, 6) as i64 - 3,
                arrival_time: Some(rng.range(0, 1000) as f64 / 1000.0),
                sampling_params: Some(SamplingParams {
                    max_tokens: Some(rng.range(1, 64) as usize),
                    ..Default::default()
                }),
                block_hasher: Some(hasher(16)),
                ..NewRequest::for_test(
                    &format!("r{next_id}"),
                    (0..num_tokens).map(|i| if i < 64 { prefix } else { i as i32 }).collect(),
                )
            })
            .unwrap();
            *next_id += 1;
            scheduler.add_request(r);
        };
        for _ in 0..10 {
            add(&mut scheduler, &mut rng, &mut next_id);
        }
        let mut finished = 0;
        for _ in 0..3000 {
            if scheduler.waiting.is_empty() && next_id < 200 {
                for _ in 0..rng.range(0, 2) {
                    add(&mut scheduler, &mut rng, &mut next_id);
                }
            }
            let output = scheduler.schedule(false);
            // Invariants.
            assert!(output.total_num_scheduled_tokens <= 8192);
            assert!(scheduler.running.len() <= 32);
            for (req_id, &n) in &output.num_scheduled_tokens {
                let r = scheduler.req(req_id);
                assert_eq!(r.status, RequestStatus::Running);
                assert!(n > 0);
                assert!(r.num_computed_tokens <= r.num_tokens() + num_spec_tokens);
            }
            let req_ids: Vec<&str> =
                output.num_scheduled_tokens.keys().map(String::as_str).collect();
            let sampled = req_ids
                .iter()
                .map(|id| {
                    let r = scheduler.req(id);
                    if r.num_computed_tokens < r.num_tokens() {
                        return vec![];
                    }
                    let n_spec = output.scheduled_spec_decode_tokens.get(*id).map_or(0, Vec::len);
                    (0..1 + rng.range(0, n_spec as u64))
                        .map(|_| rng.range(1, 30000) as i32)
                        .collect()
                })
                .collect();
            let update = scheduler.update_from_output(&output, &model_output(&req_ids, sampled));
            for eco in update.engine_core_outputs.values() {
                finished += eco.outputs.iter().filter(|o| o.finish_reason.is_some()).count();
            }
            if num_spec_tokens > 0 {
                let ids: Vec<String> = scheduler.running_ids();
                let drafts = ids
                    .iter()
                    .map(|_| (0..num_spec_tokens).map(|_| rng.range(1, 30000) as i32).collect())
                    .collect();
                scheduler.update_draft_token_ids(DraftTokenIds {
                    req_ids: ids,
                    draft_token_ids: drafts,
                });
            }
            if !scheduler.has_requests() && next_id >= 200 {
                break;
            }
        }
        assert!(!scheduler.has_unfinished_requests());
        assert_eq!(finished, next_id);
        assert_eq!(scheduler.kv_cache_manager.usage(), 0.0);
        assert_eq!(scheduler.requests.len(), 0);
    }
}
