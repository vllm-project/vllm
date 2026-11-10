// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Ports of `tests/v1/core/test_encoder_cache_manager.py`, plus a
//! differential replay of random operation sequences recorded from Python.

use serde::Deserialize;

use super::*;
use crate::sched::multimodal::{MultiModalFeatureSpec, PlaceholderRange};
use crate::sched::request::NewRequest;

struct MockRequest {
    request_id: String,
    mm_hashes: Vec<String>,
    token_counts: Vec<usize>,
}

impl MockRequest {
    fn new(request_id: &str, mm_hashes: &[&str], token_counts: &[usize]) -> Self {
        Self {
            request_id: request_id.to_owned(),
            mm_hashes: mm_hashes.iter().map(|h| (*h).to_owned()).collect(),
            token_counts: token_counts.to_vec(),
        }
    }
}

impl EncoderInputs for MockRequest {
    fn request_id(&self) -> &str {
        &self.request_id
    }

    fn mm_identifier(&self, input_id: usize) -> &str {
        &self.mm_hashes[input_id]
    }

    fn get_num_encoder_embeds(&self, input_id: usize) -> usize {
        self.token_counts[input_id]
    }

    fn num_encoder_inputs(&self) -> usize {
        self.mm_hashes.len()
    }
}

const BIG: usize = 1_000_000_000;

fn ids(manager: &EncoderCacheManager, req_id: &str) -> Option<Vec<usize>> {
    manager.request_cached_ids.get(req_id).map(|s| s.iter().collect())
}

fn referrers(manager: &EncoderCacheManager, mm_hash: &str) -> Vec<String> {
    let mut v: Vec<_> = manager.cached[mm_hash].iter().cloned().collect();
    v.sort();
    v
}

#[test]
fn test_basic_allocate_and_reuse() {
    let mut cache = EncoderCacheManager::new(10);
    let req = MockRequest::new("r1", &["imgA"], &[4]);

    assert!(!cache.check_and_update_cache(&req, 0));
    assert!(cache.can_allocate(&req, 0, BIG, 0));

    cache.allocate(&req, 0);

    assert!(cache.check_and_update_cache(&req, 0));
    assert!(cache.cached["imgA"].contains("r1"));
    assert_eq!(ids(&cache, "r1"), Some(vec![0]));
    assert_eq!(cache.num_free_slots, 6);

    // Free twice to bring refcount to 0.
    cache.free_encoder_input(&req, 0);
    cache.free_encoder_input(&req, 0);

    assert!(cache.cached["imgA"].is_empty());
    assert_eq!(ids(&cache, "r1"), None);
    assert!(cache.freeable.contains("imgA"));
    assert_eq!(cache.num_freeable_slots, 10);
    assert_eq!(cache.num_free_slots, 6);
}

#[test]
fn test_freeing_decreases_refcount_and_moves_to_freeable() {
    let mut manager = EncoderCacheManager::new(10);
    let req = MockRequest::new("req2", &["img3"], &[5]);

    assert!(manager.can_allocate(&req, 0, BIG, 0));
    manager.allocate(&req, 0);

    assert_eq!(manager.cached["img3"].len(), 1);
    assert_eq!(ids(&manager, "req2"), Some(vec![0]));

    manager.free_encoder_input(&req, 0);

    assert!(manager.cached["img3"].is_empty());
    assert_eq!(ids(&manager, "req2"), None);
    assert!(manager.freeable.contains("img3"));
    assert_eq!(manager.num_freeable_slots, 10);
}

#[test]
fn test_free_request_frees_all_inputs() {
    let mut manager = EncoderCacheManager::new(10);
    let req = MockRequest::new("req3", &["a", "b"], &[2, 3]);

    assert!(manager.can_allocate(&req, 0, BIG, 0));
    manager.allocate(&req, 0);

    assert!(manager.can_allocate(&req, 1, BIG, 0));
    manager.allocate(&req, 1);

    assert_eq!(manager.cached["a"].len(), 1);
    assert_eq!(manager.cached["b"].len(), 1);
    assert_eq!(ids(&manager, "req3"), Some(vec![0, 1]));

    manager.free(&req);

    assert!(manager.cached["a"].is_empty());
    assert!(manager.cached["b"].is_empty());
    assert_eq!(ids(&manager, "req3"), None);
    assert!(manager.freeable.contains("a"));
    assert!(manager.freeable.contains("b"));
    assert_eq!(manager.num_freeable_slots, 10);
}

#[test]
fn test_eviction_when_cache_is_full() {
    let mut manager = EncoderCacheManager::new(10);

    let req1 = MockRequest::new("req1", &["x"], &[6]);
    let req2 = MockRequest::new("req2", &["y"], &[5]);

    assert!(manager.can_allocate(&req1, 0, BIG, 0));
    manager.allocate(&req1, 0);
    manager.free_encoder_input(&req1, 0);

    assert!(manager.can_allocate(&req2, 0, BIG, 0));
    manager.allocate(&req2, 0);

    // 'x' should have been evicted.
    assert!(!manager.cached.contains_key("x"));
    assert!(!manager.num_embeds_by_hash.contains_key("x"));
    assert_eq!(ids(&manager, "req1"), None);
    assert!(manager.get_freed_mm_hashes().contains(&"x".to_owned()));
}

#[test]
fn test_get_cached_input_ids() {
    let mut manager = EncoderCacheManager::new(10);
    let req = MockRequest::new("reqX", &["m", "n", "o"], &[2, 4, 3]);

    assert!(manager.can_allocate(&req, 0, BIG, 0));
    manager.allocate(&req, 0);

    assert!(manager.can_allocate(&req, 2, BIG, 0));
    manager.allocate(&req, 2);

    assert_eq!(manager.get_cached_input_ids(&req), [0, 2]);
}

#[test]
fn test_has_cache_restores_from_freeable() {
    let mut manager = EncoderCacheManager::new(10);
    let req = MockRequest::new("reqY", &["imgZ"], &[4]);

    assert!(manager.can_allocate(&req, 0, BIG, 0));
    manager.allocate(&req, 0);

    manager.free_encoder_input(&req, 0);

    // Should restore from freeable.
    assert!(manager.check_and_update_cache(&req, 0));
    assert_eq!(manager.cached["imgZ"].len(), 1);
    assert_eq!(ids(&manager, "reqY"), Some(vec![0]));
    assert!(!manager.freeable.contains("imgZ"));
    assert_eq!(manager.num_freeable_slots, 6);
}

#[test]
fn test_get_freed_mm_hashes_clears_freed_list() {
    let mut manager = EncoderCacheManager::new(10);
    let req1 = MockRequest::new("reqA", &["a"], &[5]);
    let req2 = MockRequest::new("reqB", &["b"], &[6]);

    assert!(manager.can_allocate(&req1, 0, BIG, 0));
    manager.allocate(&req1, 0);
    manager.free_encoder_input(&req1, 0);

    // Should trigger eviction of 'a'.
    assert!(manager.can_allocate(&req2, 0, BIG, 0));
    manager.allocate(&req2, 0);

    let freed = manager.get_freed_mm_hashes();
    assert!(freed.contains(&"a".to_owned()));
    assert!(manager.get_freed_mm_hashes().is_empty());
}

/// A waiting request can hold an entry it is not scheduled for yet.
///
/// Entries with no referent are evictable. A request that is deferred (e.g.
/// by an EC connector waiting on another item) still needs the items it
/// already has, and a connector that hands out one transfer per request
/// cannot always produce the same item a second time.
#[test]
fn test_referencing_a_freeable_entry_protects_it_from_eviction() {
    let mut manager = EncoderCacheManager::new(10);
    let owner = MockRequest::new("owner", &["a"], &[5]);
    let waiter = MockRequest::new("waiter", &["a"], &[5]);
    let newcomer = MockRequest::new("newcomer", &["b"], &[6]);

    manager.allocate(&owner, 0);
    manager.free_encoder_input(&owner, 0);
    assert!(manager.freeable.contains("a"));

    // The waiting request references it; it is no longer reclaimable.
    assert!(manager.check_and_update_cache(&waiter, 0));
    assert!(!manager.freeable.contains("a"));

    // 'b' no longer fits, and 'a' must not be taken away to make room.
    assert!(!manager.can_allocate(&newcomer, 0, BIG, 0));
    assert!(manager.cached.contains_key("a"));
    assert!(manager.get_freed_mm_hashes().is_empty());

    // Once the waiter is done with it, the entry is reclaimable again.
    manager.free_encoder_input(&waiter, 0);
    assert!(manager.can_allocate(&newcomer, 0, BIG, 0));
    assert_eq!(manager.get_freed_mm_hashes(), ["a"]);
}

#[test]
fn test_reallocated_hash_is_not_reported_as_freed() {
    let mut manager = EncoderCacheManager::new(8);
    let req_a = MockRequest::new("reqA", &["a"], &[4]);
    let req_b = MockRequest::new("reqB", &["b"], &[4]);
    let req_c = MockRequest::new("reqC", &["c"], &[4]);

    manager.allocate(&req_a, 0);
    manager.allocate(&req_b, 0);
    manager.free(&req_a);
    manager.free(&req_b);

    assert!(manager.can_allocate(&req_c, 0, BIG, 0));
    manager.allocate(&req_c, 0);
    assert!(manager.can_allocate(&req_a, 0, BIG, 0));
    manager.allocate(&req_a, 0);

    assert_eq!(manager.get_freed_mm_hashes(), ["b"]);
}

#[test]
fn test_schedule_request_multi_images_respect_space_limit() {
    let mut manager = EncoderCacheManager::new(10);
    let req = MockRequest::new("reqA", &["a", "b"], &[5, 6]);
    let mut compute_budget = 100;

    let mut num_tokens_to_schedule = 0;
    assert!(manager.can_allocate(&req, 0, compute_budget, num_tokens_to_schedule));
    num_tokens_to_schedule += req.get_num_encoder_embeds(0);
    compute_budget -= req.get_num_encoder_embeds(0);

    assert!(!manager.can_allocate(&req, 1, compute_budget, num_tokens_to_schedule));
}

#[test]
fn test_schedule_request_multi_images_respect_compute_limit() {
    let mut manager = EncoderCacheManager::new(100);
    let req = MockRequest::new("reqA", &["a", "b"], &[5, 6]);
    let mut compute_budget = 10;
    let mut num_tokens_to_schedule = 0;
    assert!(manager.can_allocate(&req, 0, compute_budget, num_tokens_to_schedule));
    num_tokens_to_schedule += req.get_num_encoder_embeds(0);
    compute_budget -= req.get_num_encoder_embeds(0);

    assert!(!manager.can_allocate(&req, 1, compute_budget, num_tokens_to_schedule));
}

fn masked_request(request_id: &str, identifier: &str, is_embed: &[bool]) -> Request {
    Request::new(NewRequest {
        mm_features: vec![MultiModalFeatureSpec {
            modality: "image".into(),
            identifier: identifier.into(),
            mm_position: PlaceholderRange::new(0, is_embed.len(), Some(is_embed)),
            mm_hash: None,
            data: Default::default(),
        }],
        ..NewRequest::for_test(request_id, vec![0; is_embed.len()])
    })
    .unwrap()
}

#[test]
fn test_encoder_cache_with_is_embed_mask() {
    let mut is_embed = vec![false; 100];
    for i in [5, 15, 25, 35, 45, 55, 65, 75] {
        is_embed[i] = true;
    }
    let request = masked_request("r1", "img1", &is_embed);

    let mut manager = EncoderCacheManager::new(100);
    manager.allocate(&request, 0);

    assert_eq!(manager.num_free_slots, 92);
    assert!(manager.cached.contains_key("img1"));
    assert_eq!(ids(&manager, "r1"), Some(vec![0]));

    assert_eq!(request.mm_features[0].mm_position.get_num_embeds(), 8);
}

#[test]
fn test_encoder_cache_tracks_embedding_count_by_identifier() {
    let request = masked_request("owner", "img1", &[false, true, true, false, true, false]);

    let mut manager = EncoderCacheManager::new(10);
    manager.allocate(&request, 0);
    assert_eq!(manager.get_cached_num_encoder_embeds(&request, 0), Some(3));

    manager.free(&request);
    assert_eq!(manager.get_cached_num_encoder_embeds(&request, 0), Some(3));

    manager.reset();
    assert_eq!(manager.get_cached_num_encoder_embeds(&request, 0), None);
}

#[test]
fn test_encoder_cache_mask_based_retrieval() {
    let is_embed = [
        false, false, true, true, false, true, true, true, false, false,
    ];
    let request = masked_request("r1", "img1", &is_embed);

    let mut manager = EncoderCacheManager::new(50);
    manager.allocate(&request, 0);

    let position = &request.mm_features[0].mm_position;
    assert_eq!(position.get_num_embeds(), 5);
    // Embeddings before and within the placeholder range [2, 8) and [0, 5).
    assert_eq!(position.get_embeds_indices_in_range(2, 8), (0, 5));
    assert_eq!(position.get_embeds_indices_in_range(0, 5), (0, 2));
}

/// Test that reset() clears all cached entries and restores capacity.
#[test]
fn test_reset_clears_all_state() {
    let mut manager = EncoderCacheManager::new(20);

    let req1 = MockRequest::new("req1", &["img1", "img2"], &[5, 3]);
    let req2 = MockRequest::new("req2", &["img3"], &[4]);

    manager.allocate(&req1, 0);
    manager.allocate(&req1, 1);
    manager.allocate(&req2, 0);
    manager.free_encoder_input(&req1, 0);

    let req3 = MockRequest::new("req3", &["img4"], &[10]);
    manager.free_encoder_input(&req1, 1);
    manager.free_encoder_input(&req2, 0);
    manager.can_allocate(&req3, 0, BIG, 0);
    manager.allocate(&req3, 0);

    assert!(!manager.cached.is_empty());
    assert!(manager.num_free_slots < 20);

    manager.reset();

    assert!(manager.cached.is_empty());
    assert!(manager.num_embeds_by_hash.is_empty());
    assert!(manager.request_cached_ids.is_empty());
    assert_eq!(manager.freeable.len(), 0);
    assert!(manager.freed.is_empty());
    assert_eq!(manager.num_free_slots, 20);
    assert_eq!(manager.num_freeable_slots, 20);
}

#[test]
fn test_reset_allows_fresh_allocations() {
    let mut manager = EncoderCacheManager::new(10);

    let req1 = MockRequest::new("req1", &["img1"], &[10]);
    manager.allocate(&req1, 0);
    assert_eq!(manager.num_free_slots, 0);

    manager.reset();

    let req2 = MockRequest::new("req2", &["img2"], &[8]);
    assert!(manager.can_allocate(&req2, 0, BIG, 0));
    manager.allocate(&req2, 0);

    assert_eq!(manager.num_free_slots, 2);
    assert!(manager.cached.contains_key("img2"));
    assert!(!manager.cached.contains_key("img1"));
    assert_eq!(ids(&manager, "req2"), Some(vec![0]));
    assert_eq!(ids(&manager, "req1"), None);
}

/// Freeing a request whose two inputs share the same mm_hash must fully
/// clean up request_cached_ids. After the first free_encoder_input call,
/// cached[mm_hash] becomes empty; the second call must still remove the
/// remaining input_id from request_cached_ids.
#[test]
fn test_free_request_with_duplicate_mm_hashes() {
    let mut manager = EncoderCacheManager::new(20);

    let req = MockRequest::new("r1", &["imgA", "imgA"], &[4, 4]);

    manager.allocate(&req, 0);
    // input 1 has the same hash, so it's already cached.
    assert!(manager.check_and_update_cache(&req, 1));
    assert_eq!(ids(&manager, "r1"), Some(vec![0, 1]));

    manager.free(&req);
    assert_eq!(ids(&manager, "r1"), None);
}

/// `cached` holds one reference per request, not per position, so freeing
/// the first of two occurrences must not release the entry.
#[test]
fn test_duplicate_mm_hash_stays_referenced_until_last_free() {
    let mut manager = EncoderCacheManager::new(20);
    let req = MockRequest::new("r1", &["imgA", "imgA"], &[4, 4]);

    manager.allocate(&req, 0);
    assert!(manager.check_and_update_cache(&req, 1));

    manager.free_encoder_input(&req, 0);
    assert_eq!(referrers(&manager, "imgA"), ["r1"]);
    assert!(!manager.freeable.contains("imgA"));
    assert_eq!(manager.num_freeable_slots, 16);

    manager.free_encoder_input(&req, 1);
    assert!(manager.cached["imgA"].is_empty());
    assert_eq!(manager.freeable.get("imgA"), Some(4));
    assert_eq!(manager.num_freeable_slots, 20);
}

/// An item repeated within one request (an image carried across
/// conversation turns) must not become reclaimable capacity while a later
/// occurrence still has to be spliced in, or the encoder recomputes it.
#[test]
fn test_duplicate_mm_hash_is_not_evicted_before_last_use() {
    let mut manager = EncoderCacheManager::new(8);
    let req = MockRequest::new("r1", &["imgA", "imgA"], &[4, 4]);
    let other = MockRequest::new("r2", &["imgB"], &[8]);

    manager.allocate(&req, 0);
    assert!(manager.check_and_update_cache(&req, 1));
    manager.free_encoder_input(&req, 0);

    assert!(!manager.can_allocate(&other, 0, BIG, 0));
    assert!(manager.check_and_update_cache(&req, 1));
    assert!(manager.get_freed_mm_hashes().is_empty());
}

/// A request with many items, one repeated, releases every entry exactly
/// once its last cached occurrence is freed.
#[test]
fn test_many_inputs_release_each_hash_at_its_last_occurrence() {
    let num_items = 64;
    let mut hashes: Vec<String> = (0..num_items).map(|i| format!("img{i}")).collect();
    hashes.push("img0".into());
    let hash_refs: Vec<&str> = hashes.iter().map(String::as_str).collect();
    let mut manager = EncoderCacheManager::new(num_items);
    let req = MockRequest::new("r1", &hash_refs, &vec![1; hashes.len()]);

    for i in 0..num_items {
        manager.allocate(&req, i);
    }
    assert!(manager.check_and_update_cache(&req, num_items));

    manager.free_encoder_input(&req, 0);
    assert!(!manager.freeable.contains("img0"));
    for i in 1..num_items {
        manager.free_encoder_input(&req, i);
        assert_eq!(manager.freeable.get(&format!("img{i}")), Some(1));
    }

    manager.free_encoder_input(&req, num_items);
    assert_eq!(manager.freeable.get("img0"), Some(1));
    assert_eq!(manager.num_freeable_slots, num_items);
    assert!(!manager.request_cached_ids.contains_key("r1"));
    assert!(!manager.request_hash_refs.contains_key("r1"));
}

#[test]
fn test_encoder_decoder_cache_manager_reset() {
    let mut manager = EncoderDecoderCacheManager::new(20);

    let req1 = MockRequest::new("req1", &["img1"], &[5]);
    let req2 = MockRequest::new("req2", &["img2"], &[3]);

    manager.allocate(&req1, 0);
    manager.allocate(&req2, 0);
    manager.free(&req1);
    manager.get_freed_mm_hashes();

    assert!(manager.num_free_slots < 20);

    manager.reset();

    assert!(manager.allocated.is_empty());
    assert!(manager.to_free.is_empty());
    assert_eq!(manager.num_free_slots, 20);
}

#[test]
fn test_encoder_decoder_cache_manager_reset_allows_fresh_allocations() {
    let mut manager = EncoderDecoderCacheManager::new(10);

    let req1 = MockRequest::new("req1", &["img1"], &[10]);
    manager.allocate(&req1, 0);
    assert_eq!(manager.num_free_slots, 0);

    manager.reset();

    let req2 = MockRequest::new("req2", &["img2"], &[8]);
    assert!(manager.can_allocate(&req2, 0, BIG, 0));
    manager.allocate(&req2, 0);

    assert_eq!(manager.num_free_slots, 2);
    assert!(manager.allocated.contains(&"img2".to_owned()));
}

#[test]
fn encoder_decoder_frees_after_one_step_delay() {
    let mut manager = AnyEncoderCacheManager::EncoderDecoder(EncoderDecoderCacheManager::new(10));
    let req = MockRequest::new("r", &["a", "b"], &[2, 3]);
    assert!(!manager.check_and_update_cache(&req, 0));
    manager.allocate(&req, 0);
    manager.allocate(&req, 1);
    assert_eq!(manager.get_cached_input_ids(&req), [0, 1]);
    assert!(manager.get_freed_mm_hashes().is_empty());
    assert_eq!(manager.get_freed_mm_hashes(), ["a", "b"]);
    assert!(manager.get_freed_mm_hashes().is_empty());
}

#[test]
fn compute_budget() {
    let config = EncoderBudgetConfig {
        disable_chunked_mm_input: false,
        max_num_batched_tokens: 100,
        max_num_encoder_input_tokens: 50,
        encoder_cache_size: 80,
    };
    assert_eq!(compute_mm_encoder_budget(&config, []), Ok((0, 0)));
    assert_eq!(compute_mm_encoder_budget(&config, [10, 60]), Ok((60, 80)));
    assert_eq!(compute_mm_encoder_budget(&config, [200]), Ok((200, 200)));
    let no_chunking = EncoderBudgetConfig {
        disable_chunked_mm_input: true,
        ..config
    };
    assert!(compute_mm_encoder_budget(&no_chunking, [200]).is_err());
}

#[derive(Deserialize)]
struct Case {
    cache_size: usize,
    requests: Vec<CaseRequest>,
    ops: Vec<CaseOp>,
}

#[derive(Deserialize)]
struct CaseRequest {
    id: String,
    hashes: Vec<String>,
    counts: Vec<usize>,
}

#[derive(Deserialize)]
struct CaseOp {
    op: String,
    args: Vec<usize>,
    result: serde_json::Value,
    num_free_slots: usize,
    num_freeable_slots: usize,
    freeable: Vec<(String, usize)>,
    freed: Vec<String>,
}

/// Replays random scheduling / freeing sequences recorded from Python's
/// `EncoderCacheManager`, comparing every result and the eviction order.
#[test]
fn matches_python_encoder_cache_manager() {
    let cases: Vec<Case> = serde_json::from_str(include_str!(
        "../../../tests/fixtures/encoder_cache_cases.json"
    ))
    .unwrap();
    for (case_idx, case) in cases.iter().enumerate() {
        let reqs: Vec<MockRequest> = case
            .requests
            .iter()
            .map(|r| MockRequest {
                request_id: r.id.clone(),
                mm_hashes: r.hashes.clone(),
                token_counts: r.counts.clone(),
            })
            .collect();
        let mut manager = EncoderCacheManager::new(case.cache_size);
        for (op_idx, op) in case.ops.iter().enumerate() {
            let ctx = format!("case {case_idx} op {op_idx} ({})", op.op);
            let result = match op.op.as_str() {
                "schedule" => {
                    let (req, input_id) = (&reqs[op.args[0]], op.args[1]);
                    let outcome = if manager.check_and_update_cache(req, input_id) {
                        "hit"
                    } else if manager.can_allocate(req, input_id, op.args[2], op.args[3]) {
                        manager.allocate(req, input_id);
                        "allocated"
                    } else {
                        "rejected"
                    };
                    outcome.into()
                }
                "free_input" => {
                    manager.free_encoder_input(&reqs[op.args[0]], op.args[1]);
                    serde_json::Value::Null
                }
                "free" => {
                    manager.free(&reqs[op.args[0]]);
                    serde_json::Value::Null
                }
                "freed" => manager.get_freed_mm_hashes().into(),
                "cached_ids" => manager.get_cached_input_ids(&reqs[op.args[0]]).into(),
                other => panic!("unknown op {other}"),
            };
            assert_eq!(result, op.result, "{ctx}");
            assert_eq!(manager.num_free_slots, op.num_free_slots, "{ctx}");
            assert_eq!(manager.num_freeable_slots, op.num_freeable_slots, "{ctx}");
            let freeable: Vec<(String, usize)> =
                manager.freeable.iter().map(|(k, v)| (k.to_owned(), v)).collect();
            assert_eq!(freeable, op.freeable, "{ctx}");
            assert_eq!(manager.freed, op.freed, "{ctx}");
        }
    }
}
