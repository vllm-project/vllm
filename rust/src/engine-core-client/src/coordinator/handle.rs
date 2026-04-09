use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use zeromq::prelude::Socket;
use zeromq::{XPubSocket, XSubSocket};

use crate::coordinator::external::ExternalCoordinatorService;
use crate::coordinator::inproc::InProcCoordinatorRunner;
use crate::error::{Error, Result, bail_control_closed};
use crate::transport::EngineId;

/// Snapshot to the coordinator state for request routing and stamping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoordinatorStateSnapshot {
    /// The current DP wave, which will be stamped on outgoing requests.
    pub current_wave: u32,
    /// Whether the engines are currently running or paused, which determines if the frontend
    /// must trigger a new wave on the next request.
    pub engines_running: bool,
}

/// Shared in-process coordinator state.
pub(crate) type CoordinatorState = Mutex<CoordinatorStateSnapshot>;

/// Commands sent from the frontend request path into the background runner.
#[derive(Debug)]
pub(crate) enum CoordinatorCommand {
    /// The first request arrived while all engines were paused.
    ///
    /// The coordinator should broadcast `START_DP_WAVE` with the current wave and the target engine
    /// index as the excluded engine.
    FirstRequest {
        target_engine_id: EngineId,
        wave: u32,
    },
}

/// Frontend-facing coordinator handle used by `EngineCoreClient::call()`.
///
/// This side stays intentionally small: it can read the latest wave snapshot and
/// enqueue a `FirstRequest` transition when the request path observes the system
/// in the paused state.
#[derive(Clone)]
pub(crate) struct CoordinatorHandle {
    state: Arc<CoordinatorState>,
    command_tx: mpsc::UnboundedSender<CoordinatorCommand>,
}

impl CoordinatorHandle {
    fn new_parts() -> (
        Self,
        Arc<CoordinatorState>,
        mpsc::UnboundedReceiver<CoordinatorCommand>,
    ) {
        let state = Arc::new(Mutex::new(CoordinatorStateSnapshot {
            current_wave: 0,
            engines_running: false,
        }));
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        (
            Self {
                state: state.clone(),
                command_tx,
            },
            state,
            command_rx,
        )
    }

    /// Build the paired frontend handle and background runner around one
    /// engine-facing coordinator broadcast socket.
    pub(crate) fn new_inproc(coordinator_input: XPubSocket) -> (Self, InProcCoordinatorRunner) {
        let (handle, state, command_rx) = Self::new_parts();
        (
            handle,
            InProcCoordinatorRunner::new(state, command_rx, coordinator_input),
        )
    }

    /// Build the paired frontend handle and background service around an external
    /// Python-owned frontend-side coordinator socket.
    pub(crate) async fn connect_external(
        coordinator_address: &str,
    ) -> Result<(Self, ExternalCoordinatorService)> {
        let (handle, state, command_rx) = Self::new_parts();
        let mut socket = XSubSocket::new();
        socket.connect(coordinator_address).await?;
        socket.subscribe("").await?;
        Ok((
            handle,
            ExternalCoordinatorService::new(state, command_rx, socket),
        ))
    }

    /// Snapshot the coordinator state for request routing and stamping.
    pub(crate) fn snapshot(&self) -> CoordinatorStateSnapshot {
        *self.state.lock()
    }

    /// Notify the runner that a new request arrived while engines were paused.
    ///
    /// The handle flips `engines_running` optimistically so concurrent request
    /// submissions coalesce behind one `START_DP_WAVE` broadcast instead of all
    /// trying to trigger the wave independently.
    pub(crate) fn notify_first_request(&self, target_engine_id: EngineId) -> Result<()> {
        let mut state = self.state.lock();
        if state.engines_running {
            return Ok(());
        }

        let command = CoordinatorCommand::FirstRequest {
            target_engine_id,
            wave: state.current_wave,
        };
        if self.command_tx.send(command).is_err() {
            bail_control_closed!("in-process coordinator command channel already shut down");
        }

        state.engines_running = true;
        Ok(())
    }
}
