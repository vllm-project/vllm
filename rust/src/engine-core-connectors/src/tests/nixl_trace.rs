// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Replays scheduler traces recorded from the Python scheduler with the real
//! NIXL connector (`gen_nixl_traces.py`) against the Rust scheduler with the
//! native NIXL scheduler half, and checks that every `SchedulerOutput`,
//! `NixlConnectorMetadata`, `EngineCoreOutputs` (with the finished requests'
//! `kv_transfer_params`), KV cache usage and request count matches.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::nixl::{
    BlockIds, NixlConnectorMetadata, NixlConnectorScheduler, NixlFinishedParams, NixlKvConnector,
    NixlKvTransferParams, NixlSchedulerConfig, PyNumber, RemoteBlockIds, ReqMeta, Seq, SeqKind,
    TransferMode,
};
use vllm_engine_core::sched::output::SchedulerOutput;
use vllm_engine_core::sched::request::{FinishReason, NewRequest, SamplingParams};
use vllm_engine_core::sched::request::{Request, RequestStatus};
use vllm_engine_core::sched::scheduler::{
    AcceptAllStructuredOutput, DraftTokenIds, ErrorRequestIds, KvConnector, KvConnectorOutput,
    KvTransferConfig, KvTransferParams, ModelRunnerOutput, Scheduler,
};
use vllm_engine_core::test_util::sched_trace::{
    eco_json, first_diff, floats_match, so_json, trace_setup, usize_of,
};
use vllm_engine_core::test_util::trace_replay::replay_all;

fn ints<T: TryFrom<i64>>(v: &Value) -> Vec<T>
where
    T::Error: std::fmt::Debug,
{
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| T::try_from(x.as_i64().unwrap()).unwrap())
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
}

fn nixl_config(n: &Value) -> NixlSchedulerConfig {
    let s = |k: &str| Arc::<str>::from(n[k].as_str().unwrap());
    let i = |k: &str| n[k].as_i64().unwrap();
    let f = |k: &str| n[k].as_f64().unwrap();
    let b = |k: &str| n[k].as_bool().unwrap();
    NixlSchedulerConfig {
        mode: match n["mode"].as_str().unwrap() {
            "pull" => TransferMode::Pull,
            "push" => TransferMode::Push,
            other => panic!("unknown mode {other}"),
        },
        engine_id: s("engine_id"),
        side_channel_host: s("side_channel_host"),
        side_channel_port: i("side_channel_port"),
        transfer_tp_size: i("transfer_tp_size"),
        tensor_parallel_size: i("tensor_parallel_size"),
        decode_context_parallel_size: i("decode_context_parallel_size"),
        pipeline_parallel_size: i("pipeline_parallel_size"),
        kv_lease_duration: f("kv_lease_duration"),
        heartbeat_interval: f("heartbeat_interval"),
        use_host_buffer: b("use_host_buffer"),
        is_hma_required: b("is_hma_required"),
        num_kv_cache_groups: usize_of(&n["num_kv_cache_groups"]),
        transfer_group_ids: ints(&n["transfer_group_ids"]),
        blocks_per_sw: ints(&n["blocks_per_sw"]),
        ssm_spec_blocks: n["ssm_spec_blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize))
            .collect(),
        has_mamba: b("has_mamba"),
        bounded_replay: b("bounded_replay"),
        num_prefill_lookahead_tokens: usize_of(&n["num_prefill_lookahead_tokens"]),
        kv_recompute_threshold: i("kv_recompute_threshold"),
        bidirectional_kv_xfer: b("bidirectional_kv_xfer"),
        decoder_kv_blocks_ttl: f("decoder_kv_blocks_ttl"),
        is_kv_producer: b("is_kv_producer"),
        push_registration_timeout: f("push_registration_timeout"),
    }
}

fn list<T>(items: Vec<T>) -> Seq<T> {
    Seq::new(SeqKind::List, items)
}

/// The binding's `parse_kv_transfer_params` for the JSON the workload sends.
fn params_from_json(v: &Value) -> Option<NixlKvTransferParams> {
    let d = v.as_object()?;
    if d.is_empty() {
        return None;
    }
    let get = |k: &str| d.get(k).filter(|v| !v.is_null());
    let flag = |k: &str| get(k).is_some_and(|v| v.as_bool().unwrap());
    let int = |k: &str| get(k).map(|v| v.as_i64().unwrap());
    let string = |k: &str| get(k).map(|v| v.as_str().unwrap().to_owned());
    Some(NixlKvTransferParams {
        do_remote_prefill: flag("do_remote_prefill"),
        do_remote_decode: flag("do_remote_decode"),
        remote_block_ids: get("remote_block_ids").map(|v| {
            let items = v.as_array().unwrap();
            Arc::new(if items.first().is_some_and(Value::is_array) {
                RemoteBlockIds::Grouped(list(items.iter().map(|g| list(ints(g))).collect()))
            } else {
                RemoteBlockIds::Flat(list(ints(v)))
            })
        }),
        remote_engine_id: string("remote_engine_id").map(Arc::from),
        remote_request_id: string("remote_request_id"),
        remote_host: string("remote_host").map(Arc::from),
        remote_port: int("remote_port"),
        tp_size: int("tp_size"),
        dcp_size: int("dcp_size"),
        pp_size: int("pp_size"),
        remote_block_size: int("remote_block_size"),
        remote_num_tokens: int("remote_num_tokens"),
        remote_blocks_expiry_time: get("remote_blocks_expiry_time").map(|v| {
            if v.is_f64() {
                PyNumber::Float(v.as_f64().unwrap())
            } else {
                PyNumber::Int(v.as_i64().unwrap())
            }
        }),
        remote_registration_generation: int("remote_registration_generation"),
        p_side_truncated: flag("_p_side_truncated"),
        remote_blocks_processed: flag("_remote_blocks_processed"),
    })
}

// --- Typed JSON, as `gen_nixl_traces.tj` writes it ---

fn float(x: f64) -> Value {
    json!({ "f": x })
}

fn seq_tj(kind: SeqKind, items: Vec<Value>) -> Value {
    match kind {
        SeqKind::List => json!(items),
        SeqKind::Tuple => json!({ "t": items }),
    }
}

fn block_ids_tj(b: &BlockIds) -> Value {
    seq_tj(b.kind, b.items.iter().map(|g| json!(g)).collect())
}

fn remote_block_ids_tj(b: &RemoteBlockIds) -> Value {
    match b {
        RemoteBlockIds::Flat(s) => seq_tj(s.kind, s.items.iter().map(|x| json!(x)).collect()),
        RemoteBlockIds::Grouped(s) => seq_tj(
            s.kind,
            s.items
                .iter()
                .map(|g| seq_tj(g.kind, g.items.iter().map(|x| json!(x)).collect()))
                .collect(),
        ),
    }
}

fn number_tj(n: Option<PyNumber>) -> Value {
    match n {
        None => Value::Null,
        Some(PyNumber::Int(i)) => json!(i),
        Some(PyNumber::Float(f)) => float(f),
    }
}

fn object(class: &str, fields: Vec<(&str, Value)>) -> Value {
    json!({ "c": class, "v": fields.into_iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>() })
}

fn dict<'a>(items: impl IntoIterator<Item = (&'a str, Value)>) -> Value {
    json!({ "d": items.into_iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>() })
}

fn set<'a>(items: impl IntoIterator<Item = &'a String>) -> Value {
    let mut v: Vec<&String> = items.into_iter().collect();
    v.sort();
    json!({ "s": v })
}

fn req_meta_tj(m: &ReqMeta) -> Value {
    let remote = m.remote.as_ref().map_or(Value::Null, |r| {
        object(
            "RemoteMeta",
            vec![
                ("block_ids", remote_block_ids_tj(&r.block_ids)),
                ("host", json!(&*r.host)),
                ("port", json!(r.port)),
                ("engine_id", json!(&*r.engine_id)),
                ("request_id", json!(r.request_id)),
                ("blocks_expiry_time", number_tj(r.blocks_expiry_time)),
                ("num_tokens", json!(r.num_tokens)),
                ("registration_generation", json!(r.registration_generation)),
            ],
        )
    });
    object(
        "ReqMeta",
        vec![
            ("local_block_ids", block_ids_tj(&m.local_block_ids)),
            ("local_physical_block_ids", block_ids_tj(&m.local_block_ids)),
            ("tp_size", json!(m.tp_size)),
            ("dcp_size", json!(m.dcp_size)),
            (
                "local_num_computed_blocks",
                json!({ "t": m.local_num_computed_blocks }),
            ),
            ("remote", remote),
            ("remote_block_size", json!(m.remote_block_size)),
            ("pp_size", json!(m.pp_size)),
            ("awaiting_kvs", json!(m.awaiting_kvs)),
            ("region_blocks_to_zero", Value::Null),
            ("covered_sub_blocks", Value::Null),
        ],
    )
}

fn meta_tj(m: &NixlConnectorMetadata) -> Value {
    object(
        "NixlConnectorMetadata",
        vec![
            (
                "reqs_to_recv",
                dict(m.reqs_to_recv.iter().map(|(k, v)| (k.as_str(), req_meta_tj(v)))),
            ),
            (
                "reqs_to_save",
                dict(m.reqs_to_save.iter().map(|(k, v)| (k.as_str(), req_meta_tj(v)))),
            ),
            (
                "reqs_to_send",
                dict(m.reqs_to_send.iter().map(|(k, &v)| (k.as_str(), float(v)))),
            ),
            ("scheduler_clock", float(m.scheduler_clock)),
            ("reqs_in_batch", set(&m.reqs_in_batch)),
            ("reqs_not_processed", set(&m.reqs_not_processed)),
            (
                "heartbeat_by_engine",
                dict(m.heartbeat_by_engine.iter().map(|(engine, h)| {
                    let info = object(
                        "HeartbeatInfo",
                        vec![
                            ("req_ids", set(&h.req_ids)),
                            ("host", json!(&*h.host)),
                            ("port", json!(h.port)),
                            ("tp_size", json!(h.tp_size)),
                            ("dcp_size", json!(h.dcp_size)),
                            ("pp_size", json!(h.pp_size)),
                        ],
                    );
                    (&**engine, info)
                })),
            ),
            (
                "push_registrations",
                dict(m.push_registrations.iter().map(|(k, r)| {
                    let reg = dict([
                        ("request_id", json!(r.request_id)),
                        ("decode_engine_id", json!(&*r.decode_engine_id)),
                        ("decode_host", json!(&*r.decode_host)),
                        ("decode_port", json!(r.decode_port)),
                        ("decode_tp_size", json!(r.decode_tp_size)),
                        ("local_block_ids", block_ids_tj(&r.local_block_ids)),
                        ("remote_engine_id", json!(&*r.remote_engine_id)),
                        ("remote_host", json!(&*r.remote_host)),
                        ("remote_port", json!(r.remote_port)),
                        ("remote_tp_size", json!(r.remote_tp_size)),
                        ("remote_pp_size", json!(r.remote_pp_size)),
                    ]);
                    (k.as_str(), reg)
                })),
            ),
            (
                "push_finished_blocks",
                dict(m.push_finished_blocks.iter().map(|(k, b)| (k.as_str(), block_ids_tj(b)))),
            ),
        ],
    )
}

fn finished_tj(params: &KvTransferParams) -> Value {
    let p = params
        .params
        .0
        .downcast_ref::<NixlFinishedParams>()
        .expect("native NIXL params");
    let mut items = vec![
        ("do_remote_prefill", json!(p.do_remote_prefill)),
        ("do_remote_decode", json!(p.do_remote_decode)),
        ("remote_block_ids", json!({ "t": p.remote_block_ids })),
        ("remote_engine_id", json!(&*p.remote_engine_id)),
        ("remote_request_id", json!(p.remote_request_id)),
        ("remote_host", json!(&*p.remote_host)),
        ("remote_port", json!(p.remote_port)),
        ("tp_size", json!(p.tp_size)),
        ("dcp_size", json!(p.dcp_size)),
        ("pp_size", json!(p.pp_size)),
        ("remote_num_tokens", json!(p.remote_num_tokens)),
    ];
    if p.mode == TransferMode::Pull {
        items.push((
            "remote_blocks_expiry_time",
            p.remote_blocks_expiry_time.map_or(Value::Null, float),
        ));
        items.push((
            "remote_registration_generation",
            json!(p.remote_registration_generation),
        ));
    }
    items.push(("transfer_mode", json!(p.mode.as_str())));
    if let Some(n) = params.remote_prefill_cached_tokens {
        items.push(("remote_prefill_cached_tokens", json!(n)));
    }
    dict(items)
}

fn check(ctx: &str, what: &str, python: &Value, rust: &Value) {
    if let Some(d) = first_diff(python, rust, what) {
        panic!("{ctx}: {d}");
    }
}

fn replay(path: &Path) {
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
    let mut nixl = NixlConnectorScheduler::new(nixl_config(&c["nixl_config"])).unwrap();
    nixl.set_registration_generation(c["nixl_config"]["registration_generation"].as_u64().unwrap());
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    nixl.set_clock(Box::new(move || {
        f64::from_bits(clock.load(Ordering::Relaxed))
    }));
    let connector = NixlKvConnector::new(nixl);
    assert_eq!(
        connector.requires_kv_delivery(),
        c["requires_kv_delivery"].as_bool().unwrap()
    );
    let handle = connector.clone();
    let mut scheduler = Scheduler::new(
        sc,
        kv_cache_config,
        Some(Box::new(connector)),
        Box::new(AcceptAllStructuredOutput),
    )
    .unwrap();
    scheduler.set_clock(Box::new(|| 0.0));
    // Python finishes error requests in the iteration order of a `set` of
    // their ids (hash order); the trace shows it as the order of their
    // outputs.
    let error_order: Arc<Mutex<Vec<String>>> = Arc::default();
    let order = error_order.clone();
    scheduler.set_id_set_order(Box::new(move |ids: &ErrorRequestIds| {
        let order = order.lock().unwrap();
        let mut all: Vec<String> = ids.all().into_iter().collect();
        all.sort_by_key(|id| order.iter().position(|o| o == id).unwrap_or(usize::MAX));
        all.into_iter().collect()
    }));

    let mut outputs: Vec<SchedulerOutput> = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(1) {
        let ctx = format!("{name} line {}", i + 1);
        match line["op"].as_str().unwrap() {
            "clock" => now.store(line["t"].as_f64().unwrap().to_bits(), Ordering::Relaxed),
            "generation" => {
                handle.lock().set_registration_generation(line["n"].as_u64().unwrap());
            }
            "add" => {
                let id = line["id"].as_str().unwrap();
                let mut prompt: Vec<i32> = ints(&line["prompt"]);
                let mut params = params_from_json(&line["params"]);
                // `on_new_request`'s P-side truncation, before the request is built.
                let truncation =
                    handle.lock().prefill_truncation(params.as_ref(), prompt.len(), false);
                if let Some(backoff) = truncation {
                    prompt.truncate(prompt.len() - backoff);
                    params.as_mut().unwrap().p_side_truncated = true;
                }
                let do_remote_prefill = params.as_ref().is_some_and(|p| p.do_remote_prefill);
                handle.lock().add_request(id, params).unwrap();
                let mut request = Request::new(NewRequest {
                    request_id: id.to_owned(),
                    prompt_token_ids: Some(prompt),
                    sampling_params: Some(SamplingParams {
                        max_tokens: Some(usize_of(&line["max_tokens"])),
                        min_tokens: usize_of(&line["min_tokens"]),
                        eos_token_id: line["eos"].as_i64().map(|t| t as i32),
                        stop_token_ids: ints(&line["stop"]),
                        ..Default::default()
                    }),
                    arrival_time: Some(line["arrival_time"].as_f64().unwrap()),
                    priority: line["priority"].as_i64().unwrap(),
                    block_hasher: Some(hasher.clone()),
                    do_remote_prefill,
                    ..Default::default()
                })
                .unwrap();
                if truncation.is_some() {
                    request.max_tokens = 1;
                }
                let truncated = json!([
                    request.num_prompt_tokens,
                    request.max_tokens,
                    request.block_hashes.len()
                ]);
                check(&ctx, "truncated", &line["truncated"], &truncated);
                scheduler.add_request(request);
            }
            "abort" => {
                let ids = strings(&line["ids"]);
                let result = scheduler.finish_requests(Some(&ids), RequestStatus::FinishedAborted);
                assert_eq!(json!(result), line["result"], "{ctx}: finish_requests");
            }
            "schedule" => {
                let so = scheduler.schedule(false);
                check(&ctx, "SchedulerOutput", &line["out"], &so_json(&so));
                let meta = so.kv_connector_metadata.as_ref().expect("connector metadata");
                let meta = meta.0.downcast_ref::<NixlConnectorMetadata>().unwrap();
                check(&ctx, "NixlConnectorMetadata", &line["meta"], &meta_tj(meta));
                outputs.push(so);
            }
            "update" => {
                let so = &outputs[usize_of(&line["sched_step"])];
                let kvo = &line["kvo"];
                let mro = ModelRunnerOutput {
                    req_ids: strings(&line["req_ids"]),
                    sampled_token_ids: line["sampled"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(ints)
                        .collect(),
                    kv_connector_output: (!kvo.is_null()).then(|| KvConnectorOutput {
                        finished_recving: kvo["recving"]
                            .as_array()
                            .map(|_| strings(&kvo["recving"])),
                        finished_sending: kvo["sending"]
                            .as_array()
                            .map(|_| strings(&kvo["sending"])),
                        failed_recving: strings(&kvo["failed"]),
                        invalid_block_ids: ints(&kvo["invalid"]),
                        output: None,
                    }),
                    ..Default::default()
                };
                *error_order.lock().unwrap() = line["eco"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|client| client[1]["outputs"].as_array().unwrap())
                    .filter(|o| o[2] == json!(FinishReason::Error as u8))
                    .map(|o| o[0].as_str().unwrap().to_owned())
                    .collect();
                let update = scheduler.update_from_output(so, &mro);
                check(&ctx, "EngineCoreOutputs", &line["eco"], &eco_json(&update));
                let params: Vec<Value> = update
                    .engine_core_outputs
                    .iter()
                    .flat_map(|(_, eco)| &eco.outputs)
                    .filter_map(|o| {
                        let p = o.kv_transfer_params.as_ref()?;
                        Some(json!([o.request_id, finished_tj(p)]))
                    })
                    .collect();
                check(&ctx, "kv_transfer_params", &line["params"], &json!(params));
            }
            "draft" => {
                scheduler.update_draft_token_ids(DraftTokenIds {
                    req_ids: strings(&line["req_ids"]),
                    draft_token_ids: line["drafts"].as_array().unwrap().iter().map(ints).collect(),
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
                assert_eq!(
                    json!(handle.lock().has_pending_push_work()),
                    line["push_work"],
                    "{ctx}: has_pending_push_work"
                );
            }
            other => panic!("unknown op {other}"),
        }
    }
}

#[test]
fn matches_python_nixl_traces() {
    replay_all(
        "nixl",
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tests/nixl_traces"),
        "jsonl",
        replay,
    );
}
