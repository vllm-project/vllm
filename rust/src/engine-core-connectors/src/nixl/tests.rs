// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Scenarios from `tests/v1/kv_connector/unit/test_nixl_*.py` and
//! `test_remote_{prefill,decode}_lifecycle.py`. The trace replay
//! (`sched::scheduler::tests::nixl_differential`) checks parity with Python
//! end to end.

use std::sync::Arc;

use expect_test::expect;

use super::*;
use vllm_engine_core::sched::request::RequestStatus;

fn config(mode: TransferMode) -> NixlSchedulerConfig {
    NixlSchedulerConfig {
        mode,
        engine_id: Arc::from("local-engine"),
        side_channel_host: Arc::from("local-host"),
        side_channel_port: 5600,
        transfer_tp_size: 2,
        tensor_parallel_size: 2,
        decode_context_parallel_size: 1,
        pipeline_parallel_size: 1,
        kv_lease_duration: 30.0,
        heartbeat_interval: 5.0,
        use_host_buffer: false,
        is_hma_required: false,
        num_kv_cache_groups: 1,
        transfer_group_ids: vec![0],
        blocks_per_sw: vec![0],
        ssm_spec_blocks: vec![None],
        has_mamba: false,
        bounded_replay: false,
        num_prefill_lookahead_tokens: 0,
        kv_recompute_threshold: 64,
        bidirectional_kv_xfer: false,
        decoder_kv_blocks_ttl: 480.0,
        is_kv_producer: true,
        push_registration_timeout: 480.0,
    }
}

/// Full attention + sliding window (2 blocks kept) + Mamba with one
/// speculative slot.
fn hybrid_config(mode: TransferMode) -> NixlSchedulerConfig {
    NixlSchedulerConfig {
        is_hma_required: true,
        num_kv_cache_groups: 3,
        transfer_group_ids: vec![0, 1, 2],
        blocks_per_sw: vec![0, 2, 0],
        ssm_spec_blocks: vec![None, None, Some(1)],
        has_mamba: true,
        ..config(mode)
    }
}

fn scheduler(config: NixlSchedulerConfig) -> NixlConnectorScheduler {
    let mut s = NixlConnectorScheduler::new(config).unwrap();
    s.set_clock(Box::new(|| 100.0));
    s
}

fn d_params(remote_request_id: &str) -> NixlKvTransferParams {
    NixlKvTransferParams {
        do_remote_prefill: true,
        remote_block_ids: Some(Arc::new(RemoteBlockIds::Flat(Seq::new(
            SeqKind::List,
            vec![7, 8, 9],
        )))),
        remote_engine_id: Some(Arc::from("remote-engine")),
        remote_request_id: Some(remote_request_id.to_owned()),
        remote_host: Some(Arc::from("remote-host")),
        remote_port: Some(1234),
        tp_size: Some(1),
        ..Default::default()
    }
}

fn p_params() -> NixlKvTransferParams {
    NixlKvTransferParams {
        do_remote_decode: true,
        ..Default::default()
    }
}

fn view(request_id: &str, num_prompt_tokens: usize, status: RequestStatus) -> NixlRequestView<'_> {
    NixlRequestView {
        request_id,
        num_prompt_token_ids: num_prompt_tokens,
        num_prompt_tokens,
        num_computed_tokens: num_prompt_tokens,
        status,
    }
}

struct Blocks(Vec<Vec<BlockId>>, Vec<usize>);

impl AllocatedBlocks for Blocks {
    fn unhashed_block_ids_all_groups(&self) -> Vec<Vec<BlockId>> {
        self.0.clone()
    }

    fn num_computed_blocks(&self) -> Vec<usize> {
        self.1.clone()
    }
}

fn no_scheduled() -> std::iter::Empty<ScheduledReq<'static>> {
    std::iter::empty()
}

#[test]
fn remote_prefill_pulls_once_and_heartbeats_until_received() {
    let mut s = scheduler(config(TransferMode::Pull));
    s.add_request("d", Some(d_params("remote-d"))).unwrap();
    s.on_new_request("d");
    assert!(s.may_load_async("d"));
    assert_eq!(
        s.get_num_new_matched_tokens(&view("d", 48, RequestStatus::Waiting), 16),
        (32, true)
    );
    s.update_state_after_alloc("d", &Blocks(vec![vec![3, 4]], vec![1]), 32);
    // Only one transfer per request.
    assert!(!s.may_load_async("d"));
    assert_eq!(
        s.get_num_new_matched_tokens(&view("d", 48, RequestStatus::Waiting), 16),
        (0, false)
    );
    expect![[r#"
        NixlConnectorMetadata {
            reqs_to_recv: {
                "d": ReqMeta {
                    local_block_ids: Seq {
                        kind: Tuple,
                        items: [
                            [
                                3,
                                4,
                            ],
                        ],
                    },
                    tp_size: 1,
                    dcp_size: 1,
                    local_num_computed_blocks: [
                        1,
                    ],
                    remote: Some(
                        RemoteMeta {
                            block_ids: Flat(
                                Seq {
                                    kind: List,
                                    items: [
                                        7,
                                        8,
                                        9,
                                    ],
                                },
                            ),
                            host: "remote-host",
                            port: 1234,
                            engine_id: "remote-engine",
                            request_id: "remote-d",
                            blocks_expiry_time: None,
                            num_tokens: None,
                            registration_generation: 0,
                        },
                    ),
                    remote_block_size: None,
                    pp_size: 1,
                    awaiting_kvs: true,
                },
            },
            reqs_to_save: {},
            reqs_to_send: {},
            scheduler_clock: 100.0,
            reqs_in_batch: {},
            reqs_not_processed: {},
            heartbeat_by_engine: {
                "remote-engine": HeartbeatInfo {
                    req_ids: {
                        "remote-d",
                    },
                    host: "remote-host",
                    port: 1234,
                    tp_size: 1,
                    dcp_size: 1,
                    pp_size: 1,
                },
            },
            push_registrations: {},
            push_finished_blocks: {},
        }
    "#]]
    .assert_debug_eq(&s.build_connector_meta(no_scheduled()));
    // Throttled by heartbeat_interval.
    assert!(s.build_connector_meta(no_scheduled()).heartbeat_by_engine.is_empty());
    s.set_clock(Box::new(|| 105.0));
    s.update_connector_output(None, Some(&["d".to_owned()]));
    assert!(s.build_connector_meta(no_scheduled()).heartbeat_by_engine.is_empty());
}

#[test]
fn remote_prefill_aborted_before_scheduling_notifies_remote() {
    let mut s = scheduler(config(TransferMode::Pull));
    s.add_request("d", Some(d_params("remote-d"))).unwrap();
    let mut incomplete = d_params("remote-e");
    incomplete.remote_port = None;
    s.add_request("e", Some(incomplete)).unwrap();
    for id in ["d", "e"] {
        assert_eq!(
            s.request_finished(&view(id, 48, RequestStatus::FinishedAborted), &[vec![]]),
            (false, None)
        );
    }
    // An empty recv (a list) for "d"; "e" has no remote to notify (Python
    // raises KeyError building the metadata).
    let meta = s.build_connector_meta(no_scheduled());
    assert_eq!(meta.reqs_to_recv.keys().collect::<Vec<_>>(), ["d"]);
    assert_eq!(
        *meta.reqs_to_recv["d"].local_block_ids,
        Seq::new(SeqKind::List, vec![])
    );
    assert!(!meta.reqs_to_recv["d"].awaiting_kvs);
}

#[test]
fn producer_finish_delays_free_and_returns_clipped_blocks() {
    let mut s = scheduler(hybrid_config(TransferMode::Pull));
    s.add_request("p", Some(p_params())).unwrap();
    s.add_request("q", Some(p_params())).unwrap();
    s.update_state_after_alloc("p", &Blocks(vec![vec![]; 3], vec![0; 3]), 0);
    let block_ids = [vec![1, 2, 3], vec![0, 0, 4, 5, 6], vec![7, 8, 9]];
    let (delay, params) = s.request_finished(
        &view("p", 40, RequestStatus::FinishedLengthCapped),
        &block_ids,
    );
    assert!(delay);
    expect![[r#"
        Some(
            NixlFinishedParams {
                mode: Pull,
                do_remote_prefill: true,
                do_remote_decode: false,
                remote_block_ids: [
                    [
                        1,
                        2,
                        3,
                    ],
                    [
                        5,
                        6,
                    ],
                    [
                        8,
                    ],
                ],
                remote_engine_id: "local-engine",
                remote_request_id: "p",
                remote_host: "local-host",
                remote_port: 5600,
                tp_size: 2,
                dcp_size: 1,
                pp_size: 1,
                remote_num_tokens: 40,
                remote_blocks_expiry_time: None,
                remote_registration_generation: 0,
            },
        )
    "#]]
    .assert_debug_eq(&params);
    // An aborted producer request frees its blocks now.
    assert_eq!(
        s.request_finished(&view("q", 40, RequestStatus::FinishedAborted), &block_ids),
        (false, None)
    );
    let meta = s.build_connector_meta(no_scheduled());
    assert_eq!(
        meta.reqs_to_send.into_iter().collect::<Vec<_>>(),
        [("p".to_owned(), 130.0)]
    );
    assert_eq!(meta.reqs_in_batch.into_iter().collect::<Vec<_>>(), ["p"]);
    assert_eq!(
        meta.reqs_not_processed.into_iter().collect::<Vec<_>>(),
        ["q"]
    );
}

#[test]
fn p_side_truncation_backs_off_once() {
    let s = scheduler(hybrid_config(TransferMode::Pull));
    let mut params = p_params();
    assert_eq!(s.prefill_truncation(Some(&params), 10, false), Some(1));
    assert_eq!(s.prefill_truncation(Some(&params), 1, false), None);
    // Prompt logprobs skip the external KV: the decoder loads nothing.
    assert_eq!(s.prefill_truncation(Some(&params), 10, true), None);
    params.p_side_truncated = true;
    assert_eq!(s.prefill_truncation(Some(&params), 10, false), None);
    assert_eq!(s.prefill_truncation(Some(&d_params("r")), 10, false), None);
    // D side: the decoder recomputes the last token itself.
    let mut s = s;
    s.add_request("d", Some(d_params("remote-d"))).unwrap();
    assert_eq!(
        s.get_num_new_matched_tokens(&view("d", 10, RequestStatus::Waiting), 0),
        (9, true)
    );
}

#[test]
fn bidirectional_readback_respects_recompute_threshold() {
    let mut s = scheduler(NixlSchedulerConfig {
        bidirectional_kv_xfer: true,
        ..config(TransferMode::Pull)
    });
    let mut params = d_params("remote-p");
    params.do_remote_prefill = false;
    params.do_remote_decode = true;
    params.remote_num_tokens = Some(100);
    s.add_request("p", Some(params)).unwrap();
    let request = view("p", 80, RequestStatus::Waiting);
    // 80 - 30 = 50 tokens are below the threshold of 64.
    assert_eq!(s.get_num_new_matched_tokens(&request, 30), (0, false));
    assert_eq!(s.get_num_new_matched_tokens(&request, 0), (80, true));
    s.update_state_after_alloc("p", &Blocks(vec![vec![1]], vec![0]), 80);
    let meta = s.build_connector_meta(no_scheduled());
    assert!(meta.reqs_to_recv["p"].awaiting_kvs);
    assert!(meta.reqs_in_batch.contains("p"));
    // Processed: no second readback.
    s.update_state_after_alloc("p", &Blocks(vec![vec![1]], vec![0]), 80);
    assert!(s.build_connector_meta(no_scheduled()).reqs_to_recv.is_empty());
}

#[test]
fn push_registration_shares_blocks_with_recv_and_expires() {
    let mut s = scheduler(NixlSchedulerConfig {
        push_registration_timeout: 2.5,
        ..config(TransferMode::Push)
    });
    s.add_request("d", Some(d_params("remote-d"))).unwrap();
    s.add_request("e", Some(d_params("remote-e"))).unwrap();
    s.update_state_after_alloc("d", &Blocks(vec![vec![3, 4]], vec![0]), 32);
    assert!(s.has_pending_push_work());
    let meta = s.build_connector_meta(no_scheduled());
    assert!(!s.has_pending_push_work());
    let registration = &meta.push_registrations["d"];
    assert!(Arc::ptr_eq(
        &registration.local_block_ids,
        &meta.reqs_to_recv["d"].local_block_ids
    ));
    let remote = meta.reqs_to_recv["d"].remote.as_ref().unwrap();
    assert_eq!(*remote.block_ids, RemoteBlockIds::empty_tuple());

    // A registration not shipped before its deadline is dropped.
    s.update_state_after_alloc("e", &Blocks(vec![vec![5]], vec![0]), 16);
    s.set_clock(Box::new(|| 103.0));
    let meta = s.build_connector_meta(no_scheduled());
    assert!(meta.push_registrations.is_empty());
    assert!(meta.reqs_to_recv.contains_key("e"));
}

#[test]
fn push_producer_stages_finished_blocks_until_sent() {
    let mut s = scheduler(config(TransferMode::Push));
    s.add_request("p", Some(p_params())).unwrap();
    let (delay, params) = s.request_finished(
        &view("p", 20, RequestStatus::FinishedStopped),
        &[vec![1, 2]],
    );
    assert!(delay);
    let params = params.unwrap();
    assert!(params.do_remote_prefill && !params.do_remote_decode);
    assert!(s.has_pending_push_work());
    let meta = s.build_connector_meta(no_scheduled());
    assert_eq!(
        meta.push_finished_blocks["p"],
        Seq::new(SeqKind::Tuple, vec![vec![1, 2]])
    );
    assert!(s.has_pending_push_work());
    s.update_connector_output(Some(&["p".to_owned()]), None);
    assert!(!s.has_pending_push_work());
}

#[test]
fn push_rejects_unroutable_decode_requests() {
    let mut s = scheduler(config(TransferMode::Push));
    let mut params = d_params("remote-d");
    params.tp_size = None;
    assert!(s.add_request("d", Some(params)).is_err());
    assert_eq!(
        NixlConnectorScheduler::new(NixlSchedulerConfig {
            bidirectional_kv_xfer: true,
            ..config(TransferMode::Push)
        })
        .err(),
        Some(NixlConfigError::Unsupported(
            "Bidirectional KV transfer is not supported for NIXL push connector.".to_owned()
        ))
    );
}

#[test]
fn host_buffer_saves_until_prefill_is_scheduled() {
    let mut s = scheduler(NixlSchedulerConfig {
        use_host_buffer: true,
        ..hybrid_config(TransferMode::Pull)
    });
    s.add_request("p", Some(p_params())).unwrap();
    s.update_state_after_alloc("p", &Blocks(vec![vec![]; 3], vec![0; 3]), 0);
    let new_block_ids = [vec![1], vec![0, 2], vec![3, 4]];
    let chunk = |num_computed_tokens| ScheduledReq {
        req_id: "p",
        new_block_ids: Some(&new_block_ids),
        num_scheduled_tokens: 16,
        num_computed_tokens,
        num_prompt_tokens: 32,
    };
    let meta = s.build_connector_meta([chunk(0)]);
    // Partial prefill: SSM blocks stay unclipped for the per-step save.
    assert_eq!(
        *meta.reqs_to_save["p"].local_block_ids,
        Seq::new(SeqKind::Tuple, vec![vec![1], vec![0, 2], vec![3, 4]])
    );
    assert!(s.build_connector_meta([chunk(16)]).reqs_to_save.contains_key("p"));
    assert!(s.build_connector_meta([chunk(16)]).reqs_to_save.is_empty());
}

#[test]
fn wire_params_parse_as_the_python_dict() {
    let wire = serde_json::json!({
        "do_remote_decode": 1,
        "do_remote_prefill": "",
        "remote_block_ids": [[1, 2], [3]],
        "remote_engine_id": "engine",
        "remote_request_id": "req",
        "remote_host": "host",
        "remote_port": 5600,
        "tp_size": 2,
        "remote_blocks_expiry_time": 1.5,
        "remote_num_tokens": null,
        "unknown": {"ignored": true},
    });
    let params = NixlKvTransferParams::from_wire(&wire).unwrap().unwrap();
    assert_eq!(
        params,
        NixlKvTransferParams {
            do_remote_decode: true,
            do_remote_prefill: false,
            remote_block_ids: Some(Arc::new(RemoteBlockIds::Grouped(Seq::new(
                SeqKind::List,
                vec![
                    Seq::new(SeqKind::List, vec![1, 2]),
                    Seq::new(SeqKind::List, vec![3]),
                ],
            )))),
            remote_engine_id: Some(Arc::from("engine")),
            remote_request_id: Some("req".to_owned()),
            remote_host: Some(Arc::from("host")),
            remote_port: Some(5600),
            tp_size: Some(2),
            remote_blocks_expiry_time: Some(PyNumber::Float(1.5)),
            ..Default::default()
        }
    );
    let flat = serde_json::json!({"remote_block_ids": [4, 5], "remote_blocks_expiry_time": 7});
    let params = NixlKvTransferParams::from_wire(&flat).unwrap().unwrap();
    assert_eq!(
        params.remote_block_ids.as_deref(),
        Some(&RemoteBlockIds::Flat(Seq::new(SeqKind::List, vec![4, 5])))
    );
    assert_eq!(params.remote_blocks_expiry_time, Some(PyNumber::Int(7)));

    // Absent or empty params are none; values Python's worker would choke on
    // are rejected.
    for none in [serde_json::Value::Null, serde_json::json!({})] {
        assert_eq!(NixlKvTransferParams::from_wire(&none), Ok(None));
    }
    for invalid in [
        serde_json::json!([]),
        serde_json::json!({"remote_port": null}),
        serde_json::json!({"remote_port": "5600"}),
        serde_json::json!({"remote_block_ids": [1, "2"]}),
        serde_json::json!({"remote_blocks_expiry_time": true}),
    ] {
        assert!(
            NixlKvTransferParams::from_wire(&invalid).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn wire_request_truncates_the_p_side_prompt() {
    // Mamba: the prefiller stops one token short.
    let connector = NixlKvConnector::new(scheduler(hybrid_config(TransferMode::Pull)));
    let params = || {
        Some(NixlKvTransferParams {
            do_remote_decode: true,
            ..Default::default()
        })
    };
    let mut request = Request::new(vllm_engine_core::sched::request::NewRequest::for_test(
        "p",
        (0..9).collect(),
    ))
    .unwrap();
    connector.add_wire_request(&mut request, params(), 4, 0).unwrap();
    assert_eq!(
        request.prompt_token_ids.as_deref(),
        Some(&(0..8).collect::<Vec<_>>()[..])
    );
    assert_eq!(request.all_token_ids(), (0..8).collect::<Vec<_>>());
    assert_eq!((request.num_prompt_tokens, request.max_tokens), (8, 1));

    // Without a resolved hash block size, as Python asserts.
    let mut request = Request::new(vllm_engine_core::sched::request::NewRequest::for_test(
        "q",
        (0..9).collect(),
    ))
    .unwrap();
    assert!(connector.add_wire_request(&mut request, params(), 0, 0).is_err());

    // A decode-side request is left alone.
    let mut request = Request::new(vllm_engine_core::sched::request::NewRequest::for_test(
        "d",
        (0..9).collect(),
    ))
    .unwrap();
    connector.add_wire_request(&mut request, None, 4, 0).unwrap();
    assert_eq!(request.num_prompt_tokens, 9);
}
