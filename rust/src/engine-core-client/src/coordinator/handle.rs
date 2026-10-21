// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use parking_lot::Mutex;
use zeromq::prelude::Socket;
use zeromq::{XPubSocket, XSubSocket};

use crate::coordinator::external::ExternalCoordinatorService;
use crate::coordinator::inproc::InProcCoordinatorRunner;
use crate::error::Result;

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

    /// Build the paired frontend handle and background runner around one
    /// engine-facing coordinator broadcast socket.
    pub(crate) fn new_inproc(coordinator_input: XPubSocket) -> (Self, InProcCoordinatorRunner) {
        let (handle, state) = Self::new_parts();
        (
            handle,
            InProcCoordinatorRunner::new(state, coordinator_input),
        )
    }

    /// Build the paired frontend handle and background service around an
    /// external Python-owned frontend-side coordinator socket.
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
