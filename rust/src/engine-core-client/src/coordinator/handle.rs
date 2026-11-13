// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use parking_lot::Mutex;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use zeromq::prelude::Socket;
use zeromq::{XPubSocket, XSubSocket};

use crate::coordinator::external::ExternalCoordinatorService;
use crate::coordinator::inproc::InProcCoordinatorRunner;
use crate::error::Result;
use crate::protocol::OpaqueValue;

/// Coordinator-to-frontend state publish received on the front-side coordinator
/// socket.
///
/// This matches the msgpack tuple periodically published by Python
/// `DPCoordinatorProc.run_coordinator` to all connected frontends.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/694449050f8dac3d9853e97e518b4a43ec52106a/vllm/v1/engine/coordinator.py#L282-L283>
/// <https://github.com/vllm-project/vllm/blob/694449050f8dac3d9853e97e518b4a43ec52106a/vllm/v1/engine/coordinator.py#L445-L447>
///
/// The in-process coordinator publishes the same tuple to frontends of other
/// DP ranks.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub(super) struct CoordinatorStateUpdate {
    /// Global per-engine request counts published by the coordinator.
    ///
    /// The Rust bootstrapped external-coordinator path preserves this field for
    /// wire compatibility but intentionally ignores it for routing decisions.
    ///
    /// `None` marks a wave-only update. The in-process coordinator does not
    /// track request counts, so it always publishes `None`.
    pub counts: Option<OpaqueValue>,
    /// Current global DP wave number stamped onto newly admitted requests.
    pub wave: u32,
    /// Whether engines are currently running (`true`) or paused (`false`).
    pub engines_running: bool,
}

/// Snapshot to the coordinator state for request routing and stamping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoordinatorStateSnapshot {
    /// The current DP wave, which will be stamped on outgoing requests.
    pub current_wave: u32,
    /// Whether the engines are currently running or paused, as last reported
    /// by the engines themselves.
    pub engines_running: bool,
}

/// Shared in-process coordinator state.
pub(crate) type CoordinatorState = Mutex<CoordinatorStateSnapshot>;

/// Frontend-facing coordinator handle used by `EngineCoreClient::call()`.
///
/// The frontend only reads the latest wave snapshot: waves are started by the
/// engines, which notify the coordinator when an idle engine receives a request.
#[derive(Clone)]
pub(crate) struct CoordinatorHandle {
    state: Arc<CoordinatorState>,
}

impl CoordinatorHandle {
    fn new_parts() -> (Self, Arc<CoordinatorState>) {
        let state = Arc::new(Mutex::new(CoordinatorStateSnapshot {
            current_wave: 0,
            engines_running: false,
        }));
        (
            Self {
                state: state.clone(),
            },
            state,
        )
    }

    /// Build the paired frontend handle and background runner around the
    /// engine-facing coordinator broadcast socket and the frontend-facing state
    /// publish socket.
    pub(crate) fn new_inproc(
        coordinator_input: XPubSocket,
        frontend_publish: XPubSocket,
    ) -> (Self, InProcCoordinatorRunner) {
        let (handle, state) = Self::new_parts();
        (
            handle,
            InProcCoordinatorRunner::new(state, coordinator_input, frontend_publish),
        )
    }

    /// Build the paired frontend handle and background service around the
    /// frontend-side socket of a coordinator in another process.
    pub(crate) async fn connect_external(
        coordinator_address: &str,
    ) -> Result<(Self, ExternalCoordinatorService)> {
        let (handle, state) = Self::new_parts();
        let mut socket = XSubSocket::new();
        socket.connect(coordinator_address).await?;
        socket.subscribe("").await?;
        Ok((handle, ExternalCoordinatorService::new(state, socket)))
    }

    /// Snapshot the coordinator state for request routing and stamping.
    pub(crate) fn snapshot(&self) -> CoordinatorStateSnapshot {
        *self.state.lock()
    }
}
