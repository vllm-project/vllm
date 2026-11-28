// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Frontend admission limits: `--max-num-queued-reqs` and
//! `--max-num-queued-tokens`.
//!
//! Every request of this frontend goes through one process, so a pair of
//! atomic counters is the whole state. Python's API servers instead aggregate
//! per-process slots kept in shared memory.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use vllm_engine_core_client::RequestObserver;
use vllm_engine_core_client::protocol::output::EngineCoreOutput;

use crate::request_metrics::AdmissionRejectionCounters;
use crate::{Error, Result};

/// Queue limits checked before a request reaches engine-core. A request over
/// either one is rejected with a retryable error, served as HTTP 503 so that
/// clients and load balancers retry it elsewhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct AdmissionLimits {
    /// Maximum number of unfinished engine requests (waiting or running). A
    /// request with `n` samples takes `n`.
    pub max_num_queued_reqs: Option<u64>,
    /// Maximum total prompt tokens of requests that have not finished
    /// prefill. New requests are rejected once the backlog reaches it.
    pub max_num_queued_tokens: Option<u64>,
}

#[derive(Debug, Default)]
struct Counters {
    /// Unfinished engine requests.
    requests: AtomicU64,
    /// Prompt tokens of engine requests that have produced no output yet.
    prefill_tokens: AtomicU64,
}

/// Admission state shared by every request of one [`crate::EngineCoreLlm`].
#[derive(Debug)]
pub(crate) struct Admission {
    limits: AdmissionLimits,
    counters: Arc<Counters>,
    rejections: AdmissionRejectionCounters,
}

impl Admission {
    pub(crate) fn new(model_name: &str, limits: AdmissionLimits) -> Self {
        Self {
            limits,
            counters: Arc::default(),
            rejections: AdmissionRejectionCounters::resolve(model_name),
        }
    }

    /// Admit `n` engine requests sharing a prompt of `prompt_len` tokens,
    /// returning one guard per request.
    pub(crate) fn admit(&self, n: u32, prompt_len: u64) -> Result<Vec<AdmissionGuard>> {
        let counters = &self.counters;
        let n = u64::from(n);

        // Reserve the request slots in one step, so that concurrent admissions
        // never exceed the limit together.
        let max_reqs = self.limits.max_num_queued_reqs;
        (counters.requests)
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |requests| {
                let requests = requests + n;
                max_reqs.is_none_or(|max| requests <= max).then_some(requests)
            })
            .map_err(|_| {
                self.rejections.queued_reqs.inc();
                Error::QueueFull
            })?;

        if let Some(max) = self.limits.max_num_queued_tokens
            && counters.prefill_tokens.load(Ordering::Acquire) >= max
        {
            counters.requests.fetch_sub(n, Ordering::AcqRel);
            self.rejections.queued_tokens.inc();
            return Err(Error::PrefillBacklogFull);
        }
        counters.prefill_tokens.fetch_add(n * prompt_len, Ordering::AcqRel);

        Ok((0..n)
            .map(|_| AdmissionGuard {
                counters: counters.clone(),
                prefill_tokens: prompt_len,
            })
            .collect())
    }
}

/// One admitted engine request, owned by engine-core-client for the
/// request's lifetime. Its prompt tokens leave the prefill backlog with the
/// first output the client receives, and its request slot is freed when the
/// request leaves the client.
#[derive(Debug)]
pub(crate) struct AdmissionGuard {
    counters: Arc<Counters>,
    /// Prompt tokens still counted in the prefill backlog.
    prefill_tokens: u64,
}

impl AdmissionGuard {
    fn end_prefill(&mut self) {
        let tokens = std::mem::take(&mut self.prefill_tokens);
        if tokens > 0 {
            self.counters.prefill_tokens.fetch_sub(tokens, Ordering::AcqRel);
        }
    }
}

impl RequestObserver for AdmissionGuard {
    fn observe_output(&mut self, _output: &EngineCoreOutput) {
        // Like Python, any output from engine-core ends the request's prefill.
        self.end_prefill();
    }
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.end_prefill();
        self.counters.requests.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_limits(
        max_num_queued_reqs: Option<u64>,
        max_num_queued_tokens: Option<u64>,
    ) -> Admission {
        // The metrics registry is global: give each admission its own series.
        static NEXT_MODEL: AtomicU64 = AtomicU64::new(0);
        let model_name = format!(
            "admission-test-{}",
            NEXT_MODEL.fetch_add(1, Ordering::Relaxed)
        );
        let limits = AdmissionLimits {
            max_num_queued_reqs,
            max_num_queued_tokens,
        };
        Admission::new(&model_name, limits)
    }

    fn counts(admission: &Admission) -> (u64, u64) {
        let counters = &admission.counters;
        (
            counters.requests.load(Ordering::Acquire),
            counters.prefill_tokens.load(Ordering::Acquire),
        )
    }

    /// `vllm:admission_rejections` by `max_num_queued_reqs` and by
    /// `max_num_queued_tokens`.
    fn rejections(admission: &Admission) -> (u64, u64) {
        (
            admission.rejections.queued_reqs.get(),
            admission.rejections.queued_tokens.get(),
        )
    }

    #[test]
    fn unlimited_admission_still_counts() {
        let admission = with_limits(None, None);
        let guards = admission.admit(3, 10).unwrap();
        assert_eq!(counts(&admission), (3, 30));
        drop(guards);
        assert_eq!(counts(&admission), (0, 0));
    }

    #[test]
    fn request_limit_counts_every_sample() {
        let admission = with_limits(Some(3), None);
        let first = admission.admit(2, 1).unwrap();
        assert!(matches!(admission.admit(2, 1), Err(Error::QueueFull)));
        let second = admission.admit(1, 1).unwrap();
        assert!(matches!(admission.admit(1, 1), Err(Error::QueueFull)));
        assert_eq!(counts(&admission), (3, 3));
        // A rejection counts once, whatever its number of samples.
        assert_eq!(rejections(&admission), (2, 0));

        drop(first);
        assert!(admission.admit(2, 1).is_ok());
        drop(second);
    }

    #[test]
    fn token_limit_rejects_once_the_backlog_reaches_it() {
        let admission = with_limits(None, Some(10));
        let mut first = admission.admit(1, 6).unwrap();
        // Below the limit, a request is admitted even if its own prompt
        // takes the backlog past it.
        let second = admission.admit(1, 6).unwrap();
        assert!(matches!(
            admission.admit(1, 1),
            Err(Error::PrefillBacklogFull)
        ));
        // A rejection takes no request slot.
        assert_eq!(counts(&admission), (2, 12));
        assert_eq!(rejections(&admission), (0, 1));

        first[0].observe_output(&EngineCoreOutput::default());
        assert_eq!(counts(&admission), (2, 6));
        assert!(admission.admit(1, 1).is_ok());
        drop(second);
    }

    #[test]
    fn outputs_end_prefill_once_and_drop_frees_the_slot() {
        let admission = with_limits(Some(1), Some(10));
        let mut guard = admission.admit(1, 4).unwrap().pop().unwrap();
        guard.observe_output(&EngineCoreOutput::default());
        guard.observe_output(&EngineCoreOutput::default());
        assert_eq!(counts(&admission), (1, 0));
        drop(guard);
        assert_eq!(counts(&admission), (0, 0));
    }

    #[test]
    fn concurrent_admissions_never_exceed_the_request_limit() {
        let admission = with_limits(Some(8), None);
        let admitted: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> =
                (0..32).map(|_| scope.spawn(|| admission.admit(1, 1).ok())).collect();
            handles.into_iter().filter_map(|handle| handle.join().unwrap()).collect()
        });
        assert_eq!(admitted.len(), 8);
        assert_eq!(counts(&admission), (8, 8));
    }
}
