// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Generation readiness: whether every engine can run the model now, as
//! Python's `AsyncLLM.check_health_generate`.
//!
//! Each engine answers a `check_health_generate` utility call between steps,
//! running a forward probe when idle. A call sent to an engine cannot be taken
//! back, and a wedged engine queues every call it is sent, so each engine has
//! at most one probe in flight: callers share it, and a caller that stops
//! waiting leaves it running for the next one.

use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use parking_lot::Mutex;
use thiserror::Error;
use thiserror_ext::AsReport as _;
use tokio::sync::watch;
use tokio::time::Instant;

use crate::client::imp::ClientInner;
use crate::protocol::utility::EngineHealthState;
use crate::transport::{ConnectedEngine, EngineId};

/// Why the engines are not ready to serve generation requests.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NotReady {
    /// The engines have failed and can never serve again.
    #[error("Engine is dead")]
    EngineDead,
    /// Some engines are sleeping or paused.
    #[error("Engine is sleeping or paused (engine ranks: {engine_ranks:?})")]
    Sleeping { engine_ranks: Vec<u32> },
    /// Some engines did not answer the probe within the timeout.
    #[error("Health check timed out after {}s (engine ranks: {engine_ranks:?})", timeout.as_secs_f64())]
    ProbeTimeout {
        engine_ranks: Vec<u32>,
        timeout: Duration,
    },
    /// The probe failed on some engines; `error` describes the first failure.
    #[error("Health check failed (engine ranks: {engine_ranks:?})")]
    ProbeFailed {
        engine_ranks: Vec<u32>,
        error: String,
    },
}

type ProbeOutput = Result<EngineHealthState, String>;

/// One `check_health_generate` call to one engine. Its own task records the
/// answer, so the probe finishes, and is seen as finished, whether or not
/// anyone waits for it.
#[derive(Clone)]
struct Probe {
    /// The readiness epoch when the probe was sent: an engine answers calls in
    /// order, so a probe sent before a control call ended answers for the
    /// state before that call.
    epoch: u64,
    output: watch::Receiver<Option<ProbeOutput>>,
}

impl Probe {
    fn start(inner: &Arc<ClientInner>, engine_id: &EngineId) -> Self {
        let epoch = inner.readiness().epoch();
        let call = inner.clone();
        let engine_id = engine_id.clone();
        let (output_tx, output) = watch::channel(None);
        inner.runtime_handle().spawn(async move {
            let output = call
                .call_utility_on(&[engine_id], "check_health_generate", (), None)
                .await
                .map(|mut states: Vec<EngineHealthState>| {
                    states.pop().expect("one result per engine")
                })
                .map_err(|error| error.to_report_string());
            output_tx.send_replace(Some(output));
        });
        Self { epoch, output }
    }

    /// The probe's output, or `None` if it has none (yet).
    fn output(&self) -> Option<ProbeOutput> {
        self.output.borrow().clone()
    }

    /// Whether the probe has an output, or its task ended without one.
    fn is_finished(&self) -> bool {
        self.output.borrow().is_some() || self.output.has_changed().is_err()
    }

    async fn wait(&mut self) {
        let _ = self.output.wait_for(Option::is_some).await;
    }
}

struct EngineSlot {
    engine_id: EngineId,
    /// The data-parallel rank the engine reported in its ready response.
    rank: u32,
    /// The latest probe sent to this engine, finished or not.
    probe: Option<Probe>,
}

/// The latest probe of each connected engine.
pub(crate) struct HealthProbes {
    engines: Mutex<Vec<EngineSlot>>,
}

impl HealthProbes {
    pub fn new(engines: &[ConnectedEngine]) -> Self {
        let engines = engines
            .iter()
            .map(|engine| EngineSlot {
                engine_id: engine.engine_id.clone(),
                rank: engine.ready_response.data_parallel_rank,
                probe: None,
            })
            .collect();
        Self {
            engines: Mutex::new(engines),
        }
    }

    fn ranks(&self) -> Vec<u32> {
        self.engines.lock().iter().map(|engine| engine.rank).collect()
    }

    /// The engine's probe in flight, or a new one if its last probe finished.
    fn current_or_start(&self, inner: &Arc<ClientInner>, engine: usize) -> Probe {
        let mut engines = self.engines.lock();
        let slot = &mut engines[engine];
        if let Some(probe) = &slot.probe
            && !probe.is_finished()
        {
            return probe.clone();
        }
        let probe = Probe::start(inner, &slot.engine_id);
        slot.probe = Some(probe.clone());
        probe
    }

    /// The engine's answer to a probe sent at `epoch` or later, or `None` if
    /// none arrives before `deadline`.
    async fn answer(
        &self,
        inner: &Arc<ClientInner>,
        engine: usize,
        epoch: u64,
        deadline: Option<Instant>,
    ) -> Option<ProbeOutput> {
        loop {
            let mut probe = self.current_or_start(inner, engine);
            match deadline {
                Some(deadline) => tokio::time::timeout_at(deadline, probe.wait()).await.ok()?,
                None => probe.wait().await,
            }
            if probe.epoch >= epoch {
                return probe.output();
            }
            // A stale probe finished: ask again for the current state.
        }
    }

    /// Probe every engine, waiting at most `timeout` in total.
    pub async fn check(&self, inner: &Arc<ClientInner>, timeout: Duration) -> Result<(), NotReady> {
        if !inner.is_healthy() {
            return Err(NotReady::EngineDead);
        }
        // A call that stops serving is on its way to the engines, so they are
        // not ready whatever they answer now.
        let sleeping = || NotReady::Sleeping {
            engine_ranks: self.ranks(),
        };
        if inner.readiness().is_withdrawn() {
            return Err(sleeping());
        }
        let epoch = inner.readiness().epoch();
        // A timeout too long to represent waits forever.
        let deadline = Instant::now().checked_add(timeout);
        let engines = self.engines.lock().len();
        let answers =
            join_all((0..engines).map(|engine| self.answer(inner, engine, epoch, deadline))).await;
        if !inner.is_healthy() {
            return Err(NotReady::EngineDead);
        }
        if inner.readiness().is_withdrawn() {
            return Err(sleeping());
        }

        let mut sleeping = Vec::new();
        let mut timed_out = Vec::new();
        let mut failed = Vec::new();
        for (rank, answer) in self.ranks().into_iter().zip(answers) {
            match answer {
                None => timed_out.push(rank),
                Some(Ok(EngineHealthState::Sleeping)) => sleeping.push(rank),
                Some(Ok(EngineHealthState::Ok | EngineHealthState::Busy)) => {}
                Some(Err(error)) => failed.push((rank, error)),
            }
        }
        if !sleeping.is_empty() {
            return Err(NotReady::Sleeping {
                engine_ranks: sleeping,
            });
        }
        if !timed_out.is_empty() {
            return Err(NotReady::ProbeTimeout {
                engine_ranks: timed_out,
                timeout,
            });
        }
        if let Some((_, error)) = failed.first() {
            return Err(NotReady::ProbeFailed {
                error: error.clone(),
                engine_ranks: failed.into_iter().map(|(rank, _)| rank).collect(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::poll;
    use rmpv::Value;
    use tokio::sync::oneshot;
    use zeromq::prelude::{SocketRecv as _, SocketSend as _};
    use zeromq::{DealerSocket, PushSocket, ZmqMessage};

    use super::NotReady;
    use crate::mock_engine::default_ready_response;
    use crate::protocol::output::{EngineCoreOutputs, UtilityCallOutput};
    use crate::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
    use crate::test_utils::{
        IpcNamespace, spawn_mock_engine_task, spawn_mock_engine_task_with_ready,
    };
    use crate::{EngineCoreClient, EngineCoreClientConfig, EngineId, TransportMode};

    /// Receive one `check_health_generate` call and return its call ID.
    async fn recv_probe(dealer: &mut DealerSocket) -> u64 {
        let frames = dealer.recv().await.expect("recv utility").into_vec();
        let payload: Value = rmp_serde::from_slice(&frames[1]).expect("decode utility payload");
        let payload = payload.as_array().expect("utility payload array");
        assert_eq!(payload[2], Value::from("check_health_generate"));
        payload[1].as_u64().expect("call id")
    }

    /// Answer the probe `call_id` with `state`.
    async fn answer(push: &mut PushSocket, call_id: u64, state: &str) {
        let output = EngineCoreOutputs::from(UtilityCallOutput {
            output: UtilityOutput {
                call_id: call_id.into(),
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(Value::from(state))),
            },
            ..Default::default()
        });
        push.send(ZmqMessage::from(
            rmp_serde::to_vec_named(&output).expect("encode output"),
        ))
        .await
        .expect("send output");
    }

    /// Whether the engine receives no further message for a while.
    async fn stays_quiet(dealer: &mut DealerSocket) -> bool {
        tokio::time::timeout(Duration::from_millis(200), dealer.recv()).await.is_err()
    }

    async fn connect(
        handshake_address: String,
        ipc: &IpcNamespace,
        engines: usize,
    ) -> EngineCoreClient {
        let mut config = EngineCoreClientConfig::new_single(handshake_address)
            .with_local_input_output_addresses(
                Some(ipc.input_endpoint()),
                Some(ipc.output_endpoint()),
            );
        if let TransportMode::HandshakeOwner {
            engine_count,
            data_parallel_size,
            ..
        } = &mut config.transport_mode
        {
            *engine_count = engines;
            *data_parallel_size = engines;
        }
        EngineCoreClient::connect(config).await.expect("connect client")
    }

    fn timed_out(timeout: Duration) -> Result<(), NotReady> {
        Err(NotReady::ProbeTimeout {
            engine_ranks: vec![0],
            timeout,
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probes_engines_by_identity() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        // Identities that encode no Python engine index: each engine still
        // gets, and answers, a probe of its own, reported by its rank.
        let engines =
            [("engine-0", 0, "ok"), ("engine-1", 1, "sleeping")].map(|(id, rank, state)| {
                let mut ready = default_ready_response();
                ready.data_parallel_rank = rank;
                spawn_mock_engine_task_with_ready(
                    handshake.clone(),
                    EngineId::from(id.as_bytes().to_vec()),
                    ready,
                    move |dealer, push| {
                        Box::pin(async move {
                            let call_id = recv_probe(dealer).await;
                            answer(push, call_id, state).await;
                        })
                    },
                )
            });
        let client = connect(handshake, &ipc, 2).await;

        let result = client.check_health_generate(Duration::from_secs(5)).await;
        assert_eq!(
            result,
            Err(NotReady::Sleeping {
                engine_ranks: vec![1]
            })
        );

        for (shutdown, task) in engines {
            let _ = shutdown.send(());
            task.await.unwrap();
        }
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn callers_share_the_in_flight_probe() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (quiet_tx, quiet_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    quiet_tx.send(stays_quiet(dealer).await).unwrap();
                    answer(push, call_id, "ok").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let timeout = Duration::from_millis(50);
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        // A third caller joins the probe while the engine still holds it.
        let mut check = Box::pin(client.check_health_generate(Duration::from_secs(5)));
        assert!(poll!(check.as_mut()).is_pending());
        go_tx.send(()).unwrap();
        // The one probe every caller shares is the one the engine answers.
        assert_eq!(check.await, Ok(()));
        assert!(quiet_rx.await.unwrap(), "a second probe was sent");

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_probe_finished_after_its_caller_left_is_not_reused() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    // Answer the first probe late, as `ok`; the engine then sleeps.
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    answer(push, call_id, "ok").await;
                    let call_id = recv_probe(dealer).await;
                    answer(push, call_id, "sleeping").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let timeout = Duration::from_millis(50);
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        go_tx.send(()).unwrap();
        // The first probe's own task records the answer, without any caller.
        let mut first = client.inner.health_probes().engines.lock()[0].probe.clone().unwrap();
        first.wait().await;
        assert!(first.is_finished());

        let result = client.check_health_generate(Duration::from_secs(5)).await;
        assert_eq!(
            result,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_withdrawing_call_in_flight_reports_every_rank_sleeping() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (quiet_tx, quiet_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, _push| {
                Box::pin(async move {
                    quiet_tx.send(stays_quiet(dealer).await).unwrap();
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        // A pause on its way to the engine.
        let pause = client.inner.readiness().withdraw_until_recheck();
        let result = client.check_health_generate(Duration::from_secs(5)).await;
        assert_eq!(
            result,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );
        assert!(quiet_rx.await.unwrap(), "the engine was probed");
        drop(pause);

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_withdrawing_call_started_during_a_check_overrides_its_answer() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    // The engine answers before the pause reaches it.
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    answer(push, call_id, "ok").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let mut check = Box::pin(client.check_health_generate(Duration::from_secs(5)));
        assert!(poll!(check.as_mut()).is_pending());
        let pause = client.inner.readiness().withdraw_until_recheck();
        go_tx.send(()).unwrap();
        assert_eq!(
            check.await,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );
        drop(pause);

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_probe_sent_before_a_control_call_ended_is_replaced() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (go_tx, go_rx) = oneshot::channel();
        let (quiet_tx, quiet_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    // The first probe was answered before the control call ran,
                    // as `ok`; after the call the engine sleeps.
                    let call_id = recv_probe(dealer).await;
                    go_rx.await.unwrap();
                    quiet_tx.send(stays_quiet(dealer).await).unwrap();
                    answer(push, call_id, "ok").await;
                    let call_id = recv_probe(dealer).await;
                    answer(push, call_id, "sleeping").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;

        let timeout = Duration::from_millis(50);
        assert_eq!(
            client.check_health_generate(timeout).await,
            timed_out(timeout)
        );
        // A control call ends while the probe is in flight.
        drop(client.inner.readiness().recheck_after_call());
        // The next caller finds the stale probe still in flight.
        let mut check = Box::pin(client.check_health_generate(Duration::from_secs(5)));
        assert!(poll!(check.as_mut()).is_pending());
        go_tx.send(()).unwrap();

        assert_eq!(
            check.await,
            Err(NotReady::Sleeping {
                engine_ranks: vec![0]
            })
        );
        // The stale probe was awaited, not doubled up.
        assert!(quiet_rx.await.unwrap(), "a second probe was sent");

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }
}
