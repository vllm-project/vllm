// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde_tuple::Deserialize_tuple;
use thiserror_ext::AsReport;
use tracing::{debug, warn};
use zeromq::prelude::SocketRecv;
use zeromq::{XSubSocket, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::coordinator::handle::CoordinatorState;
use crate::error::{Error, Result, bail_unexpected_coordinator_output};
use crate::protocol::{OpaqueValue, decode_msgpack};

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
/// This owns one frontend-facing XSUB socket. It mirrors the subset of Python's
/// coordinator protocol needed by the Rust bootstrapped frontend: receive
/// `(counts, wave, running)` publishes and ignore `counts`.
pub(crate) struct ExternalCoordinatorService {
    state: Arc<CoordinatorState>,
    socket: XSubSocket,
}

impl ExternalCoordinatorService {
    pub(super) fn new(state: Arc<CoordinatorState>, socket: XSubSocket) -> Self {
        Self { state, socket }
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
                // Received publish from the external coordinator socket.
                let publish = self.socket.recv().await.map_err(Error::from)?;
                self.handle_publish(publish).await?;
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
