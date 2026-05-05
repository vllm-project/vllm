use std::sync::Arc;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport;
use tokio::sync::mpsc;
use tracing::{debug, warn};
use zeromq::prelude::SocketSend;
use zeromq::{XPubSocket, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::coordinator::handle::{CoordinatorCommand, CoordinatorState};
use crate::error::{Error, Result, bail_unexpected_coordinator_output};
use crate::protocol::{
    ClassifiedEngineCoreOutputs, DpControlMessage, EngineCoreOutputs, EngineCoreRequestType,
    encode_msgpack,
};

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
    /// Engine index that already received the triggering request and should not
    /// receive an extra wakeup notification.
    exclude_engine_index: u32,
}

/// Background half of the in-process coordinator.
///
/// This owns the command receiver and the engine-facing coordinator input
/// socket. It is the single place where wave transitions are serialized and
/// where `START_DP_WAVE` broadcasts are emitted.
pub(crate) struct InProcCoordinatorRunner {
    state: Arc<CoordinatorState>,
    command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
    coordinator_input: XPubSocket,
}

impl InProcCoordinatorRunner {
    pub(super) fn new(
        state: Arc<CoordinatorState>,
        command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
        coordinator_input: XPubSocket,
    ) -> Self {
        Self {
            state,
            command_rx,
            coordinator_input,
        }
    }

    /// Broadcast Python-compatible `START_DP_WAVE` to all connected engines.
    async fn broadcast_start_wave(&mut self, wave: u32, exclude_engine_index: u32) -> Result<()> {
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

    /// Apply one frontend-originated command to the coordinator state machine.
    async fn handle_command(&mut self, command: CoordinatorCommand) -> Result<()> {
        match command {
            CoordinatorCommand::FirstRequest {
                target_engine_id,
                wave,
            } => {
                let target_engine_index = target_engine_id.engine_index().ok_or_else(|| {
                    Error::UnsupportedCoordinatorEngineId {
                        engine_id: target_engine_id.to_vec(),
                    }
                })?;
                self.state.lock().current_wave = wave;
                debug!(
                    wave,
                    exclude_engine_index = target_engine_index,
                    "starting DP wave after first request while engines were paused"
                );
                self.broadcast_start_wave(wave, target_engine_index).await?;
            }
        }
        Ok(())
    }

    /// Apply one engine-originated control output to the coordinator state
    /// machine.
    async fn handle_outputs(&mut self, outputs: EngineCoreOutputs) -> Result<()> {
        match outputs.classify() {
            ClassifiedEngineCoreOutputs::RequestBatch(batch)
                if batch.outputs.is_empty() && batch.finished_requests.is_none() =>
            {
                // Stats-only output for coordinator.
                // Ignore since the Rust coordinator doesn't track stats for
                // routing decisions.
            }
            ClassifiedEngineCoreOutputs::DpControl {
                engine_index,
                control,
                ..
            } => match control {
                // The engines signals they completed the current wave and are now paused.
                // Advance the current wave and mark the state as paused.
                DpControlMessage::WaveComplete(wave) => {
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
                            "starting DP wave after stale-wave notification from engine"
                        );
                        self.broadcast_start_wave(wave, engine_index).await?;
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

    /// Drive the coordinator event loop until either side of the control plane
    /// is closed or a fatal error is observed.
    ///
    /// Any fatal error closes the main client registries so request streams and
    /// future calls observe a stable shutdown cause.
    pub(crate) async fn run(
        mut self,
        mut output_rx: mpsc::Receiver<Result<EngineCoreOutputs>>,
        inner: Arc<ClientInner>,
    ) {
        let result: Result<()> = async {
            loop {
                tokio::select! {
                    // Received frontend-originated command from the handle.
                    command = self.command_rx.recv() => {
                        let Some(command) = command else {
                            warn!("coordinator command channel closed, shutting down coordinator runner");
                            return Ok(());
                        };
                        self.handle_command(command).await?;
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
        }
        .await;
        let Err(error) = result else { return };

        warn!(error = %error.as_report(), "coordinator runner exiting with error");
        inner.close_registries(Arc::new(error));
    }
}
