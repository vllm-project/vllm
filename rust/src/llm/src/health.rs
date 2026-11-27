// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Generation readiness: whether every engine can run the model now, as
//! Python's `AsyncLLM.check_health_generate`.

use std::collections::HashMap;
use std::time::Duration;

use futures::future::join_all;
use parking_lot::Mutex;
use thiserror::Error;
use thiserror_ext::AsReport as _;
use tokio::sync::watch;
use vllm_engine_core_client::protocol::utility::EngineHealthState;
use vllm_engine_core_client::{EngineCoreClient, EngineId};

/// Why an [`crate::Llm`] is not ready to serve generation requests.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NotReady {
    /// The `Llm` has failed and can never serve again.
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

/// One probe of one engine, running in a task of its own: it finishes
/// whether or not anyone waits for it, and its output stays readable.
#[derive(Clone)]
struct Probe(watch::Receiver<Option<ProbeOutput>>);

impl Probe {
    fn start(client: &EngineCoreClient, engine_id: &EngineId) -> Self {
        let probe = client.check_health_generate(engine_id);
        let (output_tx, output_rx) = watch::channel(None);
        tokio::spawn(async move {
            let output = probe.await.map_err(|error| error.to_report_string());
            output_tx.send_replace(Some(output));
        });
        Self(output_rx)
    }

    /// The probe's output, or `None` while it runs.
    fn output(&self) -> Option<ProbeOutput> {
        self.0.borrow().clone()
    }

    /// Whether a new probe should replace this one: it has an output, or its
    /// task ended without one.
    fn is_finished(&self) -> bool {
        self.0.borrow().is_some() || self.0.has_changed().is_err()
    }

    async fn wait(mut self) {
        let _ = self.0.wait_for(Option::is_some).await;
    }
}

/// In-flight readiness probes per engine. Concurrent callers share one probe
/// per engine, and a caller that stops waiting leaves its probe running for
/// the next caller, so slow probes do not pile up on the engines.
#[derive(Default)]
pub(crate) struct HealthProbes {
    inflight: Mutex<HashMap<EngineId, Probe>>,
}

impl HealthProbes {
    /// Probe every engine of `client`, waiting at most `timeout` in total.
    pub(crate) async fn check(
        &self,
        client: &EngineCoreClient,
        timeout: Duration,
    ) -> Result<(), NotReady> {
        if !client.is_healthy() {
            return Err(NotReady::EngineDead);
        }
        let probes = (client.engine_ranks().into_iter())
            .map(|(engine_id, rank)| (rank, self.probe(client, &engine_id)))
            .collect::<Vec<_>>();
        // Timing out only stops waiting: the probes keep running.
        let _ = tokio::time::timeout(
            timeout,
            join_all(probes.iter().map(|(_, probe)| probe.clone().wait())),
        )
        .await;
        if !client.is_healthy() {
            return Err(NotReady::EngineDead);
        }

        let mut sleeping = Vec::new();
        let mut timed_out = Vec::new();
        let mut failed = Vec::new();
        for (engine, probe) in probes {
            match probe.output() {
                None => timed_out.push(engine),
                Some(Ok(EngineHealthState::Sleeping)) => sleeping.push(engine),
                Some(Ok(EngineHealthState::Ok | EngineHealthState::Busy)) => {}
                Some(Err(error)) => failed.push((engine, error)),
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
                engine_ranks: failed.into_iter().map(|(engine, _)| engine).collect(),
            });
        }
        Ok(())
    }

    /// Join the engine's in-flight probe, or start one if the last finished.
    fn probe(&self, client: &EngineCoreClient, engine_id: &EngineId) -> Probe {
        let mut inflight = self.inflight.lock();
        if let Some(probe) = inflight.get(engine_id)
            && !probe.is_finished()
        {
            return probe.clone();
        }
        let probe = Probe::start(client, engine_id);
        inflight.insert(engine_id.clone(), probe.clone());
        probe
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rmpv::Value;
    use tokio::sync::oneshot;
    use vllm_engine_core_client::mock_engine::default_ready_response;
    use vllm_engine_core_client::protocol::output::{EngineCoreOutputs, UtilityCallOutput};
    use vllm_engine_core_client::protocol::utility::{UtilityOutput, UtilityResultEnvelope};
    use vllm_engine_core_client::test_utils::{
        IpcNamespace, spawn_mock_engine_task, spawn_mock_engine_task_with_ready,
    };
    use vllm_engine_core_client::{
        EngineCoreClient, EngineCoreClientConfig, EngineId, TransportMode,
    };
    use zeromq::prelude::{SocketRecv as _, SocketSend as _};
    use zeromq::{DealerSocket, PushSocket, ZmqMessage};

    use super::{EngineHealthState, HealthProbes, NotReady};

    /// Receive one `check_health_generate` call and answer it with `state`.
    async fn answer_probe(dealer: &mut DealerSocket, push: &mut PushSocket, state: &str) {
        let frames = dealer.recv().await.expect("recv utility").into_vec();
        let payload: Value = rmp_serde::from_slice(&frames[1]).expect("decode utility payload");
        let payload = payload.as_array().expect("utility payload array");
        assert_eq!(payload[2], Value::from("check_health_generate"));
        let output = EngineCoreOutputs::from(UtilityCallOutput {
            output: UtilityOutput {
                call_id: payload[1].as_u64().expect("call id").into(),
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
                    move |dealer, push| Box::pin(answer_probe(dealer, push, state)),
                )
            });
        let client = connect(handshake, &ipc, 2).await;

        let result = HealthProbes::default().check(&client, Duration::from_secs(5)).await;
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
        let (answer_tx, answer_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    answer_rx.await.expect("callers timed out");
                    answer_probe(dealer, push, "ok").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;
        let probes = HealthProbes::default();
        let in_flight = || probes.inflight.lock().values().next().unwrap().0.clone();

        let timeout = Duration::from_millis(50);
        let timed_out = Err(NotReady::ProbeTimeout {
            engine_ranks: vec![0],
            timeout,
        });
        assert_eq!(probes.check(&client, timeout).await, timed_out);
        let first = in_flight();
        assert_eq!(probes.check(&client, timeout).await, timed_out);
        assert!(in_flight().same_channel(&first));

        // The one probe both callers left behind is the one the engine answers.
        answer_tx.send(()).unwrap();
        let mut first = first;
        first.wait_for(Option::is_some).await.unwrap();
        assert_eq!(*first.borrow(), Some(Ok(EngineHealthState::Ok)));

        let _ = shutdown.send(());
        task.await.unwrap();
        client.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_probe_finished_after_its_caller_left_is_not_reused() {
        let ipc = IpcNamespace::new().unwrap();
        let handshake = ipc.handshake_endpoint();
        let (answer_tx, answer_rx) = oneshot::channel();
        let (shutdown, task) =
            spawn_mock_engine_task(handshake.clone(), b"\x00\x00", |dealer, push| {
                Box::pin(async move {
                    let frames = dealer.recv().await.expect("recv utility").into_vec();
                    answer_rx.await.expect("caller timed out");
                    // Answer the first probe late, as `ok`; the engine then sleeps.
                    let payload: Value = rmp_serde::from_slice(&frames[1]).unwrap();
                    let call_id = payload.as_array().unwrap()[1].as_u64().unwrap();
                    let output = EngineCoreOutputs::from(UtilityCallOutput {
                        output: UtilityOutput {
                            call_id: call_id.into(),
                            failure_message: None,
                            result: Some(UtilityResultEnvelope::without_type_info(Value::from(
                                "ok",
                            ))),
                        },
                        ..Default::default()
                    });
                    push.send(ZmqMessage::from(rmp_serde::to_vec_named(&output).unwrap()))
                        .await
                        .unwrap();
                    answer_probe(dealer, push, "sleeping").await;
                })
            });
        let client = connect(handshake, &ipc, 1).await;
        let probes = HealthProbes::default();

        let timeout = Duration::from_millis(50);
        assert_eq!(
            probes.check(&client, timeout).await,
            Err(NotReady::ProbeTimeout {
                engine_ranks: vec![0],
                timeout
            })
        );
        answer_tx.send(()).unwrap();
        // Wait for the first probe's own task to record the answer, without any
        // caller asking for it.
        let first = probes.inflight.lock().values().next().unwrap().clone();
        first.clone().wait().await;
        assert!(first.is_finished());

        let result = probes.check(&client, Duration::from_secs(5)).await;
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
}
