// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request-completion observers, fed the same stats as the Prometheus metrics.

use std::sync::Arc;

use futures::FutureExt as _;
use futures::future::BoxFuture;

use crate::FinishReason;

/// Token counts and lifecycle timestamps of one finished request.
///
/// Timestamps are Unix seconds, or `0.0` if the event was never observed.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishedRequestStats {
    pub prompt_tokens: u32,
    pub cached_prompt_tokens: u32,
    pub generation_tokens: u32,
    /// `max_tokens` from the request's sampling params, if any.
    pub max_tokens_param: Option<u32>,
    /// The engine reported NaNs in the logits of this request.
    pub is_corrupted: bool,
    pub arrival_ts: f64,
    pub first_token_ts: f64,
    pub last_token_ts: f64,
    pub finished_ts: f64,
    /// `None` when engines do not record stats (`--disable-log-stats`).
    pub events: Option<FinishedRequestEvents>,
}

/// Request state derived from engine-core lifecycle events.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishedRequestEvents {
    pub num_preemptions: u64,
    pub queued_ts: f64,
    pub scheduled_ts: f64,
}

impl FinishedRequestStats {
    pub fn prefill_kv_computed_tokens(&self) -> u32 {
        self.prompt_tokens.saturating_sub(self.cached_prompt_tokens)
    }

    pub fn e2e_latency_seconds(&self) -> f64 {
        self.finished_ts - self.arrival_ts
    }

    pub fn queue_time_seconds(&self) -> Option<f64> {
        let events = self.events.as_ref()?;
        Some(diff_or_zero(events.scheduled_ts, events.queued_ts))
    }

    pub fn prefill_time_seconds(&self) -> Option<f64> {
        let events = self.events.as_ref()?;
        Some(diff_or_zero(self.first_token_ts, events.scheduled_ts))
    }

    pub fn decode_time_seconds(&self) -> f64 {
        diff_or_zero(self.last_token_ts, self.first_token_ts)
    }

    pub fn inference_time_seconds(&self) -> Option<f64> {
        let events = self.events.as_ref()?;
        Some(diff_or_zero(self.last_token_ts, events.scheduled_ts))
    }

    pub fn time_per_output_token_seconds(&self) -> f64 {
        if self.generation_tokens > 1 {
            self.decode_time_seconds() / (self.generation_tokens - 1) as f64
        } else {
            0.0
        }
    }
}

/// A request that reached a terminal state, as passed to [`RequestObserver`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct FinishedRequest {
    pub external_request_id: String,
    pub engine_request_id: String,
    pub model_name: String,
    pub lora_name: Option<String>,
    pub engine_index: u32,
    pub finish_reason: FinishReason,
    pub stats: FinishedRequestStats,
}

/// Consumer of finished requests.
///
/// `on_finished` runs on the request path, possibly inside `Drop`: it must not
/// block, perform I/O, or panic. Hand blocking work and I/O off to a task of
/// the observer's own.
pub trait RequestObserver: Send + Sync + 'static {
    fn on_finished(&self, request: &Arc<FinishedRequest>);

    /// Called once after the engine client has shut down, to drain pending work.
    fn shutdown(&self) -> BoxFuture<'_, ()> {
        futures::future::ready(()).boxed()
    }
}

impl<T: RequestObserver + ?Sized> RequestObserver for Arc<T> {
    fn on_finished(&self, request: &Arc<FinishedRequest>) {
        (**self).on_finished(request);
    }

    fn shutdown(&self) -> BoxFuture<'_, ()> {
        (**self).shutdown()
    }
}

/// Registered observers, notified in order. Cloning is cheap.
#[derive(Clone, Default)]
pub struct RequestObservers(Arc<[Arc<dyn RequestObserver>]>);

impl RequestObservers {
    pub fn new(observers: Vec<Arc<dyn RequestObserver>>) -> Self {
        Self(observers.into())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn notify(&self, request: FinishedRequest) {
        let request = Arc::new(request);
        for observer in self.0.iter() {
            observer.on_finished(&request);
        }
    }

    pub(crate) async fn shutdown(&self) {
        for observer in self.0.iter() {
            observer.shutdown().await;
        }
    }
}

pub(crate) struct ObserverContext {
    pub(crate) observers: RequestObservers,
    pub(crate) external_request_id: String,
    pub(crate) engine_request_id: String,
    pub(crate) model_name: String,
    pub(crate) lora_name: Option<String>,
    pub(crate) engine_index: u32,
}

impl ObserverContext {
    pub(crate) fn notify(self, finish_reason: FinishReason, stats: FinishedRequestStats) {
        let Self {
            observers,
            external_request_id,
            engine_request_id,
            model_name,
            lora_name,
            engine_index,
        } = self;
        observers.notify(FinishedRequest {
            external_request_id,
            engine_request_id,
            model_name,
            lora_name,
            engine_index,
            finish_reason,
            stats,
        });
    }
}

fn diff_or_zero(end: f64, start: f64) -> f64 {
    if end > 0.0 && start > 0.0 && end >= start {
        end - start
    } else {
        0.0
    }
}
