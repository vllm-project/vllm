// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Replays KV cache manager traces recorded from the Python implementation
//! (`gen_traces.py`) and checks every result matches.

use std::collections::HashMap;
use std::fmt::Write as _;

use super::TestRequest;
use crate::hash::BlockHash;
use crate::kv_cache::*;

fn fmt_groups(blocks: Option<&[Vec<BlockId>]>) -> String {
    let Some(blocks) = blocks else {
        return "None".to_owned();
    };
    blocks
        .iter()
        .map(|g| g.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join("|")
}

fn fmt_hash(h: Option<&ExternalBlockHash>) -> String {
    match h {
        None => "-".to_owned(),
        Some(ExternalBlockHash::Int(i)) => i.to_string(),
        Some(ExternalBlockHash::Bytes(b)) => {
            b.as_bytes()[..8.min(b.0.len())].iter().fold(String::new(), |mut s, x| {
                write!(s, "{x:02x}").unwrap();
                s
            })
        }
    }
}

/// Python `repr` of `list[list[int]] | None`.
fn fmt_lookahead(la: &Option<Vec<Vec<i32>>>) -> String {
    match la {
        None => "None".to_owned(),
        Some(v) => format!(
            "[{}]",
            v.iter()
                .map(|t| format!(
                    "[{}]",
                    t.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn fmt_events(events: &[KvCacheEvent]) -> String {
    events
        .iter()
        .map(|e| match e {
            KvCacheEvent::BlockStored(e) => format!(
                "S g={} h={} p={} t={} bs={} k={} w={} la={}",
                e.group_idx.map_or("None".to_owned(), |g| g.to_string()),
                e.block_hashes.iter().map(|h| fmt_hash(Some(h))).collect::<Vec<_>>().join(","),
                fmt_hash(e.parent_block_hash.as_ref()),
                e.token_ids.len(),
                e.block_size,
                e.kv_cache_spec_kind.unwrap_or("None"),
                e.kv_cache_spec_sliding_window.map_or("None".to_owned(), |w| w.to_string()),
                fmt_lookahead(&e.lookahead_token_ids),
            ),
            KvCacheEvent::BlockRemoved(e) => format!(
                "R g={} h={}",
                e.group_idx.map_or("None".to_owned(), |g| g.to_string()),
                fmt_hash(e.block_hashes.first()),
            ),
            KvCacheEvent::AllBlocksCleared(_) => "C".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Python `repr` of `list[tuple[int, int, int]]`.
fn fmt_tuples(v: &[BoundaryStateOffload]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|(a, b, c)| format!("({a}, {b}, {c})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn parse_spec(args: &[&str]) -> KvCacheSpec {
    let n = |i: usize| args[i].parse::<usize>().unwrap();
    let bs = n(1);
    let spec_type = match args[0] {
        "full" => KvCacheSpecType::FullAttention,
        "mla" => KvCacheSpecType::MlaAttention,
        "swa" => KvCacheSpecType::SlidingWindow {
            sliding_window: n(2),
            extra_retained_tokens: n(3),
        },
        "swa_mla" => KvCacheSpecType::SlidingWindowMla {
            sliding_window: n(2),
            extra_retained_tokens: 0,
            bounded_replay: n(3) == 1,
        },
        "local" => KvCacheSpecType::ChunkedLocalAttention {
            attention_chunk_size: n(2),
        },
        "mamba" => KvCacheSpecType::Mamba {
            mamba_cache_mode: match args[2] {
                "align" => MambaCacheMode::Align,
                "all" => MambaCacheMode::All,
                _ => MambaCacheMode::None,
            },
            num_speculative_blocks: n(3),
            num_prefill_checkpoint_blocks: n(4),
            prefill_checkpoint_alignment: args[5].parse().ok(),
        },
        other => panic!("unknown spec {other}"),
    };
    KvCacheSpec::new(spec_type, bs)
}

fn parse_list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    if s == "-" {
        return Vec::new();
    }
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

fn parse_hex(s: &str) -> BlockHash {
    let bytes: Vec<u8> = (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect();
    BlockHash::from_slice(&bytes)
}

/// Replays one trace, returning the first mismatch.
fn replay(name: &str, text: &str) -> Result<usize, String> {
    let mut lines = text.lines().enumerate().peekable();
    let (_, config_line) = lines.next().unwrap();
    let kv: HashMap<&str, &str> = config_line
        .split_whitespace()
        .skip(1)
        .map(|kv| kv.split_once('=').unwrap())
        .collect();
    let num = |k: &str| kv[k].parse::<usize>().unwrap();
    let mut groups = Vec::new();
    while let Some((_, line)) = lines.peek()
        && line.starts_with("group ")
    {
        let args: Vec<&str> = line.split_whitespace().skip(1).collect();
        groups.push(KvCacheGroupSpec::new(parse_spec(&args)));
        lines.next();
    }
    let mut config = KvCacheConfig::new(num("num_blocks"), groups);
    config.prefix_cache_retention_interval = kv["retention"].parse().ok();
    config.needs_kv_cache_zeroing = num("zeroing") == 1;
    let mut manager_config = KvCacheManagerConfig::new(
        num("max_model_len"),
        num("scheduler_block_size"),
        num("hash_block_size"),
    );
    manager_config.enable_caching = num("enable_caching") == 1;
    manager_config.use_eagle = num("use_eagle") == 1;
    manager_config.num_hash_lookahead_tokens = num("num_hash_lookahead_tokens");
    manager_config.enable_kv_cache_events = num("events") == 1;
    manager_config.kv_events_use_int_block_hashes = num("int_hashes") == 1;
    manager_config.log_stats = true;
    manager_config.dcp_world_size = num("dcp");
    manager_config.enable_mamba_shared_prefix_checkpoint = num("shared_ckpt") == 1;
    manager_config.max_in_flight_tokens =
        kv.get("max_in_flight_tokens").and_then(|v| v.parse().ok());
    manager_config.num_prefill_lookahead =
        kv.get("num_prefill_lookahead").map_or(0, |v| v.parse().unwrap());
    let mut manager = KvCacheManager::new(config, manager_config).unwrap();

    let mut reqs: HashMap<String, TestRequest> = HashMap::new();
    let mut last_hit: HashMap<String, KvCacheBlocks> = HashMap::new();
    let mut num_checks = 0;
    while let Some((lineno, line)) = lines.next() {
        let args: Vec<&str> = line.split_whitespace().collect();
        let actual = match args[0] {
            "step" => {
                manager.new_step_starts();
                continue;
            }
            "sync" => {
                let req = reqs.entry(args[1].to_owned()).or_insert_with(|| {
                    let mut req = super::make_request(args[1], Vec::new(), 1);
                    req.waiting = true;
                    req
                });
                req.num_prompt_tokens = args[2].parse().unwrap();
                req.num_computed_tokens = args[3].parse().unwrap();
                req.shared_prefix_boundary = args[4].parse().unwrap();
                req.tokens.extend(parse_list::<i32>(args[5]));
                if args[6] != "-" {
                    req.block_hashes.extend(args[6].split(',').map(parse_hex));
                }
                continue;
            }
            "free" => {
                manager.free(args[1]);
                continue;
            }
            "record" => {
                manager.record_prefix_cache_stats(&reqs[args[1]], args[2].parse().unwrap());
                continue;
            }
            "get" => {
                let (blocks, n, boundary) = manager.get_computed_blocks(&reqs[args[1]], None);
                let s = format!("{n} {boundary} {}", fmt_groups(Some(&blocks.blocks)));
                last_hit.insert(args[1].to_owned(), blocks);
                s
            }
            "alloc" => {
                let p = |i: usize| args.get(i).map_or(0, |a| a.parse::<usize>().unwrap());
                let computed = (args[5] == "1").then(|| last_hit[args[1]].clone());
                let out = manager
                    .allocate_slots(
                        &reqs[args[1]],
                        p(2),
                        AllocateSlotsArgs {
                            num_new_computed_tokens: p(3),
                            new_computed_blocks: computed.as_ref(),
                            num_lookahead_tokens: p(4),
                            num_external_computed_tokens: p(6),
                            full_sequence_must_fit: p(7) == 1,
                            delay_cache_blocks: p(8) == 1,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                fmt_groups(out.as_ref().map(|b| b.blocks.as_slice()))
            }
            "cache" => {
                manager.cache_blocks(&reqs[args[1]], args[2].parse().unwrap());
                continue;
            }
            "getc" => {
                let (blocks, n, boundary, diverged) =
                    manager.get_computed_blocks_for_connector(&reqs[args[1]], None);
                let s = format!(
                    "{n} {boundary} {} {}",
                    u8::from(diverged),
                    fmt_groups(Some(&blocks.blocks))
                );
                last_hit.insert(args[1].to_owned(), blocks);
                s
            }
            "finalize" => fmt_tuples(&manager.finalize_partial_tail_offloads(&reqs[args[1]])),
            "estimate" => manager.estimate_cached_tokens(&reqs[args[1]]).to_string(),
            "cached" => manager.get_num_cached_tokens(&reqs[args[1]]).to_string(),
            "ids_computed" => fmt_groups(Some(
                &manager.get_block_ids_for_computed_tokens(args[1], args[2].parse().unwrap()),
            )),
            "copies" => {
                let (copies, retained) = manager.take_kv_cache_block_copies();
                let s = format!(
                    "{} {}",
                    copies
                        .iter()
                        .map(|c| format!("{}>{}", c.src_block_id, c.dst_block_id))
                        .collect::<Vec<_>>()
                        .join(","),
                    retained.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
                );
                manager.block_pool_mut().free_blocks(retained, false);
                s
            }
            "offloads" => manager
                .take_boundary_state_offloads()
                .iter()
                .map(|(k, v)| format!("{k}:{}", fmt_tuples(v)))
                .collect::<Vec<_>>()
                .join(";"),
            "zero" => manager
                .take_new_block_ids()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            "evict" => {
                manager.evict_blocks(parse_list::<BlockId>(args[1]));
                continue;
            }
            "blocks" => fmt_groups(Some(&manager.get_blocks(args[1]).blocks)),
            "state" => {
                let free = manager.block_pool().free_block_queue.get_all_free_blocks();
                format!(
                    "{} {:.6}",
                    free.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
                    manager.usage()
                )
            }
            "events" => fmt_events(&manager.take_events()),
            "reset" => if manager.reset_prefix_cache() {
                "True"
            } else {
                "False"
            }
            .to_owned(),
            "stats" => {
                let s = manager.make_prefix_cache_stats().unwrap();
                format!("{} {} {}", s.requests, s.queries, s.hits)
            }
            other => panic!("{name}:{}: unknown op {other}", lineno + 1),
        };
        let (_, expected) = lines.next().unwrap();
        let expected = expected.strip_prefix("= ").unwrap_or(expected.trim_start_matches('='));
        if actual != expected {
            return Err(format!(
                "{name}:{}: `{line}`\n  python: {expected}\n  rust:   {actual}",
                lineno + 1
            ));
        }
        num_checks += 1;
    }
    Ok(num_checks)
}

#[test]
fn replay_python_traces() {
    let traces = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kv_cache/tests/traces");
    crate::test_util::trace_replay::replay_all("kv_cache", &traces, "trace", |path| {
        let name = path.file_name().unwrap().to_string_lossy();
        let text = std::fs::read_to_string(path).unwrap();
        match replay(&name, &text) {
            Ok(n) => eprintln!("{name}: {n} results match"),
            Err(e) => panic!("{e}"),
        }
    });
}
