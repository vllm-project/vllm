// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Whether the connected engines can accept and process new requests.
//!
//! The engines own this state and report it through the `is_inference_ready`
//! utility. The client withdraws readiness while a call that stops serving is
//! in flight, and rechecks after every control call ends, asking the engines
//! again since only they know what the call left behind. Readiness therefore
//! becomes `true` only from an engine answer.
//!
//! Withdrawing readiness publishes `false`, which the gRPC `readiness` and
//! `vllm.Inference` health services report as `NOT_SERVING`, the same
//! withdrawal the server publishes on shutdown.
//!
//! Asking runs an RPC on every worker, so the client asks only while someone
//! subscribes; a new subscriber triggers an ask if the answer is unknown.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use thiserror_ext::AsReport as _;
use tokio::sync::{Notify, watch};
use tracing::warn;

use crate::client::imp::ClientInner;

/// How long to wait before asking the engines again after a failed attempt.
const ASK_RETRY_DELAY: Duration = Duration::from_secs(1);

#[derive(Debug)]
struct ReadinessState {
    /// Calls that stop serving and have not ended yet; readiness stays
    /// withdrawn while any is in flight.
    withdrawals: usize,
    /// The engines' latest answer, or `None` from the end of a control call
    /// until they answer again.
    engines_ready: Option<bool>,
    /// Advanced whenever a control call ends, so an answer that may predate
    /// the call is discarded.
    epoch: u64,
}

impl ReadinessState {
    /// Readiness to publish, or `None` to keep the published value until the
    /// engines answer.
    fn ready(&self) -> Option<bool> {
        if self.withdrawals > 0 {
            Some(false)
        } else {
            self.engines_ready
        }
    }
}

/// Whether the connected engines can accept and process new requests, as
/// published to subscribers. Control calls report through [`RecheckGuard`]
/// guards, and [`run_readiness_loop`] records the engines' answers.
pub(crate) struct Readiness {
    state: Mutex<ReadinessState>,
    tx: watch::Sender<bool>,
    /// Wakes the ask loop when an answer may be needed.
    ask: Notify,
}

impl Readiness {
    /// Start ready: engines without sleep mode cannot start paused, and
    /// [`Readiness::start`] records the engines' answer for those with it.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(ReadinessState {
                withdrawals: 0,
                engines_ready: Some(true),
                epoch: 0,
            }),
            tx: watch::Sender::new(true),
            ask: Notify::new(),
        }
    }

    /// Subscribe to the published readiness, asking the engines if their
    /// answer is unknown.
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        let rx = self.tx.subscribe();
        self.ask.notify_one();
        rx
    }

    /// For a control call that may change readiness, such as a resume: leave
    /// readiness unchanged during the call and recheck after it ends. Hold the
    /// guard until the call ends.
    pub fn recheck_after_call(&self) -> RecheckGuard<'_> {
        RecheckGuard {
            readiness: self,
            withdraws: false,
        }
    }

    /// For a control call that stops serving, such as a pause: withdraw
    /// readiness from before the call is sent until the recheck after it ends.
    /// Hold the guard until the call ends.
    pub fn withdraw_until_recheck(&self) -> RecheckGuard<'_> {
        let mut state = self.state.lock();
        state.withdrawals += 1;
        self.publish(&state);
        RecheckGuard {
            readiness: self,
            withdraws: true,
        }
    }

    /// Record the engines' answer at startup, before any control call.
    pub fn start(&self, ready: bool) {
        let mut state = self.state.lock();
        state.engines_ready = Some(ready);
        self.publish(&state);
    }

    /// Record the engines' answer to an ask made at `epoch`, unless a control
    /// call ended since.
    fn answer(&self, epoch: u64, ready: bool) {
        let mut state = self.state.lock();
        if state.epoch == epoch {
            state.engines_ready = Some(ready);
            self.publish(&state);
        }
    }

    /// Return the epoch to ask at while the engines' answer is unknown and
    /// someone subscribes.
    fn pending_ask(&self) -> Option<u64> {
        let state = self.state.lock();
        (state.engines_ready.is_none() && self.tx.receiver_count() > 0).then_some(state.epoch)
    }

    fn end(&self, withdraws: bool) {
        let mut state = self.state.lock();
        if withdraws {
            state.withdrawals -= 1;
        }
        state.epoch += 1;
        state.engines_ready = None;
        self.ask.notify_one();
        self.publish(&state);
    }

    fn publish(&self, state: &ReadinessState) {
        if let Some(ready) = state.ready() {
            self.tx.send_if_modified(|current| std::mem::replace(current, ready) != ready);
        }
    }
}

/// A control call in flight. Dropping it, whether the call succeeded, failed,
/// or was cancelled, rechecks: the client asks the engines again.
pub(crate) struct RecheckGuard<'a> {
    readiness: &'a Readiness,
    /// Whether readiness is withdrawn until the recheck.
    withdraws: bool,
}

impl Drop for RecheckGuard<'_> {
    fn drop(&mut self) {
        self.readiness.end(self.withdraws);
    }
}

/// Background loop that asks the engines whenever their answer is unknown,
/// until an answer arrives with no control call ending in between.
pub(crate) async fn run_readiness_loop(inner: Arc<ClientInner>) {
    let readiness = inner.readiness();
    loop {
        let Some(epoch) = readiness.pending_ask() else {
            readiness.ask.notified().await;
            continue;
        };
        match inner.is_inference_ready().await {
            Ok(ready) => readiness.answer(epoch, ready),
            // Engine failure is permanent and fails every health service.
            Err(_) if !inner.is_healthy() => return,
            Err(error) => {
                warn!(
                    error = %error.as_report(),
                    "failed to ask the engines whether they are ready; keeping the published readiness"
                );
                tokio::time::sleep(ASK_RETRY_DELAY).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_that_predates_a_control_call_is_discarded() {
        let readiness = Readiness::new();
        let published = readiness.subscribe();

        drop(readiness.recheck_after_call()); // resume
        let before_pause = readiness.pending_ask().expect("ask after the resume");
        drop(readiness.withdraw_until_recheck()); // pause
        let after_pause = readiness.pending_ask().expect("ask after the pause");

        // Answered before the pause reached the engines.
        readiness.answer(before_pause, true);
        assert!(!*published.borrow());
        assert_eq!(readiness.pending_ask(), Some(after_pause));

        readiness.answer(after_pause, false);
        assert!(!*published.borrow());
        assert_eq!(readiness.pending_ask(), None);
    }

    #[test]
    fn withdrawal_in_flight_hides_engine_answers() {
        let readiness = Readiness::new();
        let published = readiness.subscribe();

        let pause = readiness.withdraw_until_recheck();
        assert!(!*published.borrow());
        // A resume that overtakes the pause.
        drop(readiness.recheck_after_call());
        let epoch = readiness.pending_ask().unwrap();
        readiness.answer(epoch, true);
        assert!(!*published.borrow());

        drop(pause);
        assert!(!*published.borrow());
        let epoch = readiness.pending_ask().unwrap();
        readiness.answer(epoch, true);
        assert!(*published.borrow());
    }

    #[test]
    fn asks_only_while_subscribed() {
        let readiness = Readiness::new();
        drop(readiness.recheck_after_call());
        assert_eq!(readiness.pending_ask(), None);

        let _published = readiness.subscribe();
        assert!(readiness.pending_ask().is_some());
    }
}
