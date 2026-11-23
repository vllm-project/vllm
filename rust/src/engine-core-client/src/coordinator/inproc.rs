// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport;
use tokio::sync::{Notify, mpsc};
use tracing::{debug, warn};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{XPubSocket, ZmqError, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::coordinator::bootstrap::is_subscribe_all;
use crate::coordinator::handle::{CoordinatorState, CoordinatorStateUpdate};
use crate::error::{Error, Result, bail_unexpected_coordinator_output};
use crate::protocol::encode_msgpack;
use crate::protocol::output::{DpControlMessage, DpControlOutput, EngineCoreOutputs};
use crate::protocol::request::EngineCoreRequestType;

/// Coordinator-to-engine `START_DP_WAVE` control payload encoded on the
/// engine-facing coordinator socket.
///
/// This matches the msgpack tuple broadcast by Python
/// `DPCoordinatorProc._send_start_wave`.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/694449050f8dac3d9853e97e518b4a43ec52106a/vllm/v1/engine/coordinator.py#L453-L459>
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
struct StartDpWaveMessage {
    /// DP wave number that all engines should start processing.
    wave: u32,
    /// Engine index that started the wave and so does not need an extra wakeup.
    /// `None` wakes every engine.
    exclude_engine_index: Option<u32>,
}

/// Background half of the in-process coordinator.
///
/// This owns the engine-facing coordinator input socket. It is the single place
/// where wave transitions are serialized and where `START_DP_WAVE` broadcasts
/// are emitted.
pub(crate) struct InProcCoordinatorRunner {
    state: Arc<CoordinatorState>,
    coordinator_input: XPubSocket,
    /// Wakes the frontend publisher after each wave transition.
    state_changed: Arc<Notify>,
    /// Moved out by `run` to run beside the engine-facing loop.
    frontend_publisher: Option<FrontendStatePublisher>,
}

/// Frontend-facing half of the in-process coordinator, through which frontends
/// of other DP ranks follow the wave state that this process reads in memory.
///
/// It runs beside the engine-facing loop, so a frontend that stops reading can
/// delay frontend updates but never engine wave control. Updates coalesce to
/// the latest state, which is all a frontend reads.
struct FrontendStatePublisher {
    state: Arc<CoordinatorState>,
    socket: XPubSocket,
    state_changed: Arc<Notify>,
}

impl FrontendStatePublisher {
    /// Publish the current wave state to subscribed frontends.
    async fn publish_state(&mut self) -> Result<()> {
        let state = *self.state.lock();
        let payload = encode_msgpack(&CoordinatorStateUpdate {
            counts: None,
            wave: state.current_wave,
            engines_running: state.engines_running,
        })?;
        self.socket.send(ZmqMessage::from(payload)).await?;
        Ok(())
    }

    /// Publish on every wave transition and to every new subscriber until a
    /// fatal socket error.
    async fn run(mut self) -> Result<()> {
        loop {
            tokio::select! {
                () = self.state_changed.notified() => self.publish_state().await?,
                // As on the engine-facing socket, zmq.rs registers a subscription only
                // when `recv()` yields it. Answering it with the current state covers a
                // frontend that subscribes after the last wave transition.
                subscription = self.socket.recv() => {
                    match subscription {
                        Ok(message) => {
                            if is_subscribe_all(&message) {
                                self.publish_state().await?;
                            }
                        }
                        Err(error @ (ZmqError::Codec(_) | ZmqError::Network(_))) => {
                            warn!(
                                error = %error,
                                "coordinator frontend subscription peer disconnected"
                            );
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
    }
}

impl InProcCoordinatorRunner {
    pub(super) fn new(
        state: Arc<CoordinatorState>,
        coordinator_input: XPubSocket,
        frontend_publish: XPubSocket,
    ) -> Self {
        let state_changed = Arc::new(Notify::new());
        Self {
            frontend_publisher: Some(FrontendStatePublisher {
                state: state.clone(),
                socket: frontend_publish,
                state_changed: state_changed.clone(),
            }),
            state,
            coordinator_input,
            state_changed,
        }
    }

    /// Broadcast Python-compatible `START_DP_WAVE` to all connected engines.
    async fn broadcast_start_wave(
        &mut self,
        wave: u32,
        exclude_engine_index: Option<u32>,
    ) -> Result<()> {
        let payload = encode_msgpack(&StartDpWaveMessage {
            wave,
            exclude_engine_index,
        })?;
        self.coordinator_input
            .send(
                ZmqMessage::try_from(vec![
                    EngineCoreRequestType::StartDpWave.to_frame(),
                    payload.into(),
                ])
                .expect("coordinator START_DP_WAVE message must contain two frames"),
            )
            .await?;
        Ok(())
    }

    /// Apply one engine-originated control output to the coordinator state
    /// machine.
    async fn handle_outputs(&mut self, outputs: EngineCoreOutputs) -> Result<()> {
        match outputs {
            EngineCoreOutputs::RequestBatch(batch)
                if batch.outputs.is_empty() && batch.finished_requests.is_none() =>
            {
                // Stats-only output for coordinator.
                // Ignore since the Rust coordinator doesn't track stats for
                // routing decisions.
            }
            EngineCoreOutputs::DpControl(DpControlOutput {
                engine_index,
                control,
                ..
            }) => match control {
                // The engines signals they completed the current wave and are now paused.
                // Advance the current wave and mark the state as paused.
                DpControlMessage::WaveComplete(wave) => {
                    let advanced = {
                        let mut state = self.state.lock();
                        if wave >= state.current_wave {
                            let next_wave = wave + 1;
                            debug!(
                                wave,
                                next_wave,
                                "DP wave finished; pausing engines and advancing coordinator state"
                            );
                            state.current_wave = wave + 1;
                            state.engines_running = false;
                            true
                        } else {
                            false
                        }
                    };
                    if advanced {
                        self.state_changed.notify_one();
                    }
                }
                // An engine requests to start the wave.
                // Rebroadcast the wave to all engines except for the originated one.
                DpControlMessage::StartWave(wave) => {
                    let should_broadcast = {
                        let mut state = self.state.lock();
                        if wave > state.current_wave
                            || (wave == state.current_wave && !state.engines_running)
                        {
                            state.current_wave = wave;
                            state.engines_running = true;
                            true
                        } else {
                            false
                        }
                    };
                    if should_broadcast {
                        debug!(
                            wave,
                            exclude_engine_index = engine_index,
                            "starting DP wave on notification from engine"
                        );
                        self.broadcast_start_wave(wave, Some(engine_index)).await?;
                        self.state_changed.notify_one();
                    }
                }
            },
            other => {
                bail_unexpected_coordinator_output!(
                    "received non-control output on coordinator path: {other:?}"
                );
            }
        }
        Ok(())
    }

    /// Drive the coordinator event loop until the engine control output is
    /// closed or a fatal error is observed.
    ///
    /// Any fatal error closes the main client registries so request streams and
    /// future calls observe a stable shutdown cause.
    pub(crate) async fn run(
        mut self,
        mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
        inner: Arc<ClientInner>,
    ) {
        let frontend_publisher =
            self.frontend_publisher.take().expect("the coordinator runner runs once");
        let engine_loop = async {
            loop {
                tokio::select! {
                    // Keep receiving engine subscriptions: zmq.rs XPUB registers a
                    // subscription only when `recv()` yields it (unlike libzmq), so an
                    // engine whose XSUB reconnects gets no broadcasts until we read its
                    // subscribe-all. If a wave is running, replay it so that engine does
                    // not miss the `START_DP_WAVE` sent while it was disconnected.
                    subscription = self.coordinator_input.recv() => {
                        match subscription {
                            Ok(message) => {
                                if is_subscribe_all(&message) {
                                    let state = *self.state.lock();
                                    if state.engines_running {
                                        debug!(
                                            current_wave = state.current_wave,
                                            "replaying active DP wave after engine subscription"
                                        );
                                        self.broadcast_start_wave(state.current_wave, None).await?;
                                    }
                                }
                            }
                            Err(error @ (ZmqError::Codec(_) | ZmqError::Network(_))) => {
                                warn!(
                                    error = %error,
                                    "coordinator subscription peer disconnected"
                                );
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                    // Received engine-originated control output from the coordinator socket.
                    outputs = output_rx.recv() => {
                        let Some(outputs) = outputs else {
                            warn!("coordinator output channel closed, shutting down coordinator runner");
                            return Ok(());
                        };
                        self.handle_outputs(outputs?).await?;
                    }
                }
            }
        };
        let result: Result<()> = tokio::select! {
            result = engine_loop => result,
            result = frontend_publisher.run() => result,
        };
        let Err(error) = result else { return };

        warn!(error = %error.as_report(), "coordinator runner exiting with error");
        inner.close_registries(Arc::new(error));
    }
}
