use std::sync::Arc;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport;
use tokio::sync::mpsc;
use tracing::{debug, warn};
use zeromq::prelude::{SocketRecv, SocketSend};
use zeromq::{XSubSocket, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::coordinator::handle::{CoordinatorCommand, CoordinatorState};
use crate::error::{Error, Result, bail_unexpected_coordinator_output};
use crate::protocol::{OpaqueValue, decode_msgpack, encode_msgpack};

/// Frontend-to-coordinator wakeup message sent when the first request arrives
/// while all engines are paused.
///
/// This matches the frontend-side msgpack tuple sent by Python
/// `DPAsyncMPClient._ensure_stats_update_task` to the coordinator front socket.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/694449050f8dac3d9853e97e518b4a43ec52106a/vllm/v1/engine/core_client.py#L1230-L1236>
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
struct CoordinatorWakeupMessage {
    /// Engine index that already has the triggering request and should be
    /// excluded from the coordinator's `START_DP_WAVE` rebroadcast.
    exclude_engine_index: u32,
    /// DP wave number observed by the frontend when the request was admitted.
    wave: u32,
}

/// Coordinator-to-frontend state publish received on the front-side coordinator
/// socket.
///
/// This matches the msgpack tuple periodically published by Python
/// `DPCoordinatorProc.run_coordinator` to all connected frontends.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/694449050f8dac3d9853e97e518b4a43ec52106a/vllm/v1/engine/coordinator.py#L282-L283>
/// <https://github.com/vllm-project/vllm/blob/694449050f8dac3d9853e97e518b4a43ec52106a/vllm/v1/engine/coordinator.py#L445-L447>
#[derive(Debug, Clone, PartialEq, Deserialize_tuple)]
struct CoordinatorStateUpdate {
    /// Global per-engine request counts published by the coordinator.
    ///
    /// The Rust bootstrapped external-coordinator path preserves this field for
    /// wire compatibility but intentionally ignores it for routing decisions.
    counts: OpaqueValue,
    /// Current global DP wave number stamped onto newly admitted requests.
    wave: u32,
    /// Whether engines are currently running (`true`) or paused (`false`).
    engines_running: bool,
}

/// Background half of an external Python-owned coordinator connection.
///
/// This owns the command receiver and one frontend-facing XSUB socket. It
/// mirrors the subset of Python's coordinator protocol needed by the Rust
/// bootstrapped frontend: receive `(counts, wave, running)` publishes, ignore
/// `counts`, and send `(exclude_engine_index, wave)` wakeup messages when the
/// first request arrives while engines are paused.
pub(crate) struct ExternalCoordinatorService {
    state: Arc<CoordinatorState>,
    command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
    socket: XSubSocket,
}

impl ExternalCoordinatorService {
    pub(super) fn new(
        state: Arc<CoordinatorState>,
        command_rx: mpsc::UnboundedReceiver<CoordinatorCommand>,
        socket: XSubSocket,
    ) -> Self {
        Self {
            state,
            command_rx,
            socket,
        }
    }

    /// Apply one frontend-originated command to the external coordinator state
    /// machine.
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
                debug!(
                    wave,
                    exclude_engine_index = target_engine_index,
                    "notifying external coordinator about first request while engines were paused"
                );
                let payload = encode_msgpack(&CoordinatorWakeupMessage {
                    exclude_engine_index: target_engine_index,
                    wave,
                })?;
                self.socket.send(ZmqMessage::from(payload)).await?;
            }
        }
        Ok(())
    }

    /// Apply one publish received from the xsub socket containing a coordinator
    /// state update.
    async fn handle_publish(&mut self, message: ZmqMessage) -> Result<()> {
        let frames = message.into_vec();
        if frames.len() != 1 {
            bail_unexpected_coordinator_output!(
                "received malformed external coordinator publish with {} frame(s)",
                frames.len()
            );
        }

        let update: CoordinatorStateUpdate = decode_msgpack(&frames[0])?;

        let mut state = self.state.lock();
        let previous_wave = state.current_wave;
        let previous_engines_running = state.engines_running;
        state.current_wave = update.wave;
        state.engines_running = update.engines_running;
        debug!(
            previous_wave,
            wave = update.wave,
            previous_engines_running,
            engines_running = update.engines_running,
            "applied external coordinator state update"
        );
        Ok(())
    }

    /// Drive the coordinator event loop until either side of the control plane
    /// is closed or a fatal error is observed.
    pub(crate) async fn run(mut self, inner: Arc<ClientInner>) {
        let result: Result<()> = async {
            loop {
                tokio::select! {
                    // Received frontend-originated command from the handle.
                    command = self.command_rx.recv() => {
                        let Some(command) = command else {
                            warn!("external coordinator command channel closed, shutting down service");
                            return Ok(());
                        };
                        self.handle_command(command).await?;
                    }
                    // Received publish from the external coordinator socket.
                    publish = self.socket.recv() => {
                        let publish = publish.map_err(Error::from)?;
                        self.handle_publish(publish).await?;
                    }
                }
            }
        }
        .await;
        let Err(error) = result else { return };

        warn!(
            error = %error.as_report(),
            "external coordinator service exiting with error"
        );
        inner.close_registries(Arc::new(error));
    }
}
