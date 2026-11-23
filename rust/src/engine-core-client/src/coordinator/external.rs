// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use thiserror_ext::AsReport;
use tracing::{debug, warn};
use zeromq::prelude::SocketRecv;
use zeromq::{XSubSocket, ZmqMessage};

use crate::client::imp::ClientInner;
use crate::coordinator::handle::{CoordinatorState, CoordinatorStateUpdate};
use crate::error::{Error, Result, bail_unexpected_coordinator_output};
use crate::protocol::decode_msgpack;

/// Background half of a connection to a coordinator in another process: a
/// Python coordinator, or the in-process coordinator of another Rust frontend.
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

    /// Drive the coordinator event loop until the task is aborted or a fatal
    /// error is observed.
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
