// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of `vllm/v1/core/sched/utils.py`.

use std::collections::HashSet;
use std::hash::{BuildHasher, Hash};

use super::request::{RepetitionDetectionParams, Request, RequestStatus, StopReason};

/// Check if the tail of token_ids contains a repeating pattern.
///
/// Compares the last pattern_len tokens against the preceding
/// (repetition_min_count - 1) repetitions of the same length.
fn has_repeating_pattern(
    token_ids: &[i32],
    pattern_len: usize,
    repetition_min_count: usize,
) -> bool {
    let len = token_ids.len();
    for n in 1..=pattern_len {
        let target_token = token_ids[len - n];
        for m in 1..repetition_min_count {
            if token_ids[len - (pattern_len * m + n)] != target_token {
                return false;
            }
        }
    }
    true
}

/// Check if a sequence of token IDs has a repetition pattern.
///
/// Args:
///     token_ids: List of token IDs
///     params: Repetition detection parameters.
///
/// Returns:
///     True if a repetition pattern is found, False otherwise.
pub fn check_sequence_repetition(token_ids: &[i32], params: &RepetitionDetectionParams) -> bool {
    let max_pattern_size = params.max_pattern_size;
    let mut min_pattern_size = params.min_pattern_size;
    let min_count = params.min_count;

    if min_pattern_size == 0 {
        min_pattern_size = 1;
    }

    if max_pattern_size == 0 || min_count < 2 || min_pattern_size > max_pattern_size {
        return false;
    }

    for pattern_len in min_pattern_size..=max_pattern_size {
        if pattern_len * min_count > token_ids.len() {
            return false;
        }

        if has_repeating_pattern(token_ids, pattern_len, min_count) {
            return true;
        }
    }

    false
}

/// Remove all items from a list that are in the items_to_remove set.
///
/// This method optimizes for the common case of removing a single item,
/// falling back to list comprehension for multiple items.
///
/// Note: like Python, a single item is removed only at its first occurrence,
/// while multiple items are removed at every occurrence. Python returns the
/// list (the same one or a new one); this filters `lst` in place.
pub fn remove_all<T: Eq + Hash, S: BuildHasher>(lst: &mut Vec<T>, items_to_remove: &HashSet<T, S>) {
    if items_to_remove.is_empty() {
        return;
    }

    if items_to_remove.len() == 1 {
        // Fast path for single item removal (most common case)
        let item = items_to_remove.iter().next().unwrap();
        if let Some(pos) = lst.iter().position(|x| x == item) {
            lst.remove(pos);
        }
        return;
    }
    // For multiple items, use list comprehension
    lst.retain(|item| !items_to_remove.contains(item));
}

pub fn check_stop(request: &mut Request, max_model_len: usize) -> bool {
    assert!(request.pooling_params.is_none());

    let sampling_params = request.sampling_params.as_ref().expect("sampling_params is None");

    let last_token_id = *request.output_token_ids().last().expect("request has no output tokens");
    if Some(last_token_id) == sampling_params.eos_token_id {
        request.status = RequestStatus::FinishedStopped;
        return true;
    }

    if sampling_params.stop_token_ids.contains(&last_token_id) {
        request.status = RequestStatus::FinishedStopped;
        request.stop_reason = Some(StopReason::TokenId(last_token_id as u32));
        return true;
    }

    if request.num_tokens() >= max_model_len || request.num_output_tokens() >= request.max_tokens {
        request.status = RequestStatus::FinishedLengthCapped;
        return true;
    }

    // Note(arpera):
    // Order of checks is important for min_tokens
    // If you decide to change the order, first look at these PRs please:
    // 47489, 49521, and 51299
    // These PRs show some reasoning about the order of checks
    if request.num_output_tokens() < sampling_params.min_tokens {
        return false;
    }

    if let Some(repetition_detection) = &sampling_params.repetition_detection
        && check_sequence_repetition(request.output_token_ids(), repetition_detection)
    {
        request.status = RequestStatus::FinishedRepetition;
        request.stop_reason = Some(StopReason::Text("repetition_detected".to_owned()));
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use super::*;
    use crate::sched::request::{NewRequest, PoolingParams, SamplingParams};

    fn params(
        max_pattern_size: usize,
        min_pattern_size: usize,
        min_count: usize,
    ) -> RepetitionDetectionParams {
        RepetitionDetectionParams {
            max_pattern_size,
            min_pattern_size,
            min_count,
        }
    }

    // Ports of `TestCheckSequenceRepetition` in
    // `tests/v1/core/test_repetition_detection.py`.
    #[test]
    fn check_sequence_repetition_cases() {
        let cases: &[(&[i32], RepetitionDetectionParams, bool)] = &[
            // test_simple_repetition_detected
            (&[1, 2, 3, 1, 2, 3, 1, 2, 3], params(3, 2, 3), true),
            // test_repetition_below_min_count
            (&[1, 2, 3, 1, 2, 3], params(3, 2, 3), false),
            // test_two_token_pattern
            (&[1, 2, 1, 2, 1, 2, 1, 2], params(5, 2, 4), true),
            // test_no_repetition_varied_sequence
            (&[1, 2, 3, 4, 5, 6, 7, 8, 9], params(5, 2, 2), false),
            // test_partial_repetition_not_detected
            (&[1, 2, 3, 1, 2, 3, 1, 2, 4], params(3, 2, 3), false),
            // test_empty_token_list
            (&[], params(3, 2, 2), false),
            // test_detection_disabled_max_size_zero
            (&[1, 2, 1, 2, 1, 2], params(0, 0, 0), false),
            // test_invalid_min_count
            (&[1, 2, 1, 2], params(0, 0, 0), false),
            (&[1, 2, 1, 2], params(2, 0, 1), false),
            // test_repetition_at_end_of_sequence
            (&[1, 2, 3, 4, 5, 6, 5, 6, 5, 6], params(3, 2, 3), true),
            // min_pattern_size 0 defaults to 1.
            (&[4, 7, 7], params(3, 0, 2), true),
            // A too-long pattern stops the search at the first such size.
            (&[1, 1], params(3, 2, 2), false),
        ];
        for (i, (tokens, params, expected)) in cases.iter().enumerate() {
            assert_eq!(
                check_sequence_repetition(tokens, params),
                *expected,
                "case {i}"
            );
        }
        // test_large_pattern_many_repetitions
        let tokens: Vec<i32> = [1, 2, 3, 4, 5, 6, 7, 8].repeat(5);
        assert!(check_sequence_repetition(&tokens, &params(10, 2, 3)));
    }

    fn request(sp: SamplingParams, prompt: Vec<i32>, output: &[i32]) -> Request {
        let mut r = Request::new(NewRequest {
            sampling_params: Some(sp),
            ..NewRequest::for_test("test", prompt)
        })
        .unwrap();
        r.append_output_token_ids(output);
        r
    }

    fn with_repetition(min_tokens: usize, rd: RepetitionDetectionParams) -> SamplingParams {
        SamplingParams {
            max_tokens: Some(100),
            min_tokens,
            repetition_detection: Some(rd),
            ..Default::default()
        }
    }

    // Ports of `TestRepetitionDetectionIntegration`.
    #[test]
    fn check_stop_repetition_integration() {
        // test_basic_repetition_stops_generation
        let mut r = request(
            with_repetition(0, params(5, 2, 3)),
            vec![1, 2, 3],
            &[10, 20, 10, 20, 10, 20],
        );
        assert!(check_stop(&mut r, 1024));
        assert_eq!(r.status, RequestStatus::FinishedRepetition);
        assert_eq!(
            r.stop_reason,
            Some(StopReason::Text("repetition_detected".into()))
        );

        // test_detection_disabled_no_stop
        let sp = SamplingParams {
            max_tokens: Some(100),
            ..Default::default()
        };
        let mut r = request(sp, vec![1, 2, 3], &[10, 20, 10, 20, 10, 20]);
        assert!(!check_stop(&mut r, 1024));

        // test_repetition_respects_min_tokens
        let mut r = request(
            with_repetition(10, params(5, 2, 3)),
            vec![1, 2, 3],
            &[10, 20, 10, 20, 10, 20],
        );
        assert!(!check_stop(&mut r, 1024));

        // test_no_repetition_continues_generation
        let mut r = request(
            with_repetition(0, params(5, 2, 3)),
            vec![1, 2, 3],
            &[10, 20, 30, 40, 50, 60],
        );
        assert!(!check_stop(&mut r, 1024));
        assert_eq!(r.status, RequestStatus::Waiting);

        // test_pattern_at_size_boundary
        let mut r = request(
            with_repetition(0, params(3, 3, 2)),
            vec![1, 2],
            &[10, 20, 30, 10, 20, 30],
        );
        assert!(check_stop(&mut r, 1024));
        assert_eq!(r.status, RequestStatus::FinishedRepetition);

        // test_multiple_pattern_sizes_checked
        let mut r = request(
            with_repetition(0, params(5, 2, 3)),
            vec![1],
            &[7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10],
        );
        assert!(check_stop(&mut r, 1024));
        assert_eq!(r.status, RequestStatus::FinishedRepetition);

        // test_eos_takes_precedence_over_repetition
        let sp = SamplingParams {
            stop_token_ids: vec![999],
            ..with_repetition(0, params(5, 2, 3))
        };
        let mut r = request(sp, vec![1, 2, 3], &[10, 20, 10, 20, 999]);
        assert!(check_stop(&mut r, 1024));
        assert_eq!(r.status, RequestStatus::FinishedStopped);

        // test_min_pattern_size_filters_small_patterns
        let mut r = request(
            with_repetition(0, params(5, 3, 3)),
            vec![1],
            &[10, 20, 10, 20, 10, 20],
        );
        assert!(!check_stop(&mut r, 1024));

        // test_high_repetition_threshold
        let mut r = request(
            with_repetition(0, params(5, 2, 5)),
            vec![1],
            &[10, 20, 10, 20, 10, 20],
        );
        assert!(!check_stop(&mut r, 1024));
    }

    #[test]
    fn check_stop_order() {
        let base = SamplingParams {
            max_tokens: Some(4),
            min_tokens: 3,
            eos_token_id: Some(2),
            stop_token_ids: vec![7],
            ..Default::default()
        };

        // EOS stops before min_tokens without a stop reason.
        let mut r = request(base.clone(), vec![1], &[2]);
        assert!(check_stop(&mut r, 100));
        assert_eq!(
            (r.status, r.stop_reason.clone()),
            (RequestStatus::FinishedStopped, None)
        );

        // Stop tokens record the token as the stop reason.
        let mut r = request(base.clone(), vec![1], &[7]);
        assert!(check_stop(&mut r, 100));
        assert_eq!(r.stop_reason, Some(StopReason::TokenId(7)));

        // EOS in the prompt or earlier in the output is irrelevant.
        let mut r = request(base.clone(), vec![2], &[2, 5]);
        assert!(!check_stop(&mut r, 100));

        // max_tokens.
        let mut r = request(base.clone(), vec![1], &[5, 5, 5, 5]);
        assert!(check_stop(&mut r, 100));
        assert_eq!(r.status, RequestStatus::FinishedLengthCapped);

        // max_model_len counts prompt tokens.
        let mut r = request(base.clone(), vec![1, 1, 1], &[5]);
        assert!(check_stop(&mut r, 4));
        assert_eq!(r.status, RequestStatus::FinishedLengthCapped);

        // ignore_eos clears `eos_token_id`.
        let ignore_eos = SamplingParams {
            eos_token_id: None,
            ..base
        };
        let mut r = request(ignore_eos, vec![1], &[2]);
        assert!(!check_stop(&mut r, 100));
    }

    #[test]
    #[should_panic]
    fn check_stop_rejects_pooling() {
        let mut r = Request::new(NewRequest {
            pooling_params: Some(PoolingParams::default()),
            ..NewRequest::for_test("r", vec![1])
        })
        .unwrap();
        r.append_output_token_ids(&[1]);
        check_stop(&mut r, 10);
    }

    #[test]
    fn remove_all_semantics() {
        let set = |items: &[i32]| items.iter().copied().collect::<FxHashSet<_>>();

        let mut lst = vec![1, 2, 1, 3];
        remove_all(&mut lst, &set(&[]));
        assert_eq!(lst, [1, 2, 1, 3]);
        // A single item is removed at its first occurrence only.
        remove_all(&mut lst, &set(&[1]));
        assert_eq!(lst, [2, 1, 3]);
        remove_all(&mut lst, &set(&[9]));
        assert_eq!(lst, [2, 1, 3]);

        let mut lst = vec![1, 2, 1, 3];
        remove_all(&mut lst, &set(&[1, 3]));
        assert_eq!(lst, [2]);
    }
}
