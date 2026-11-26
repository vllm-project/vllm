// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use thiserror_ext::AsReport;
use tracing::{debug, warn};
use zeromq::prelude::SocketRecv;
use zeromq::{XSubSocket, ZmqError, ZmqMessage};

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
                let publish = match self.socket.recv().await {
                    Ok(publish) => publish,
                    Err(error @ (ZmqError::Codec(_) | ZmqError::Network(_))) => {
                        warn!(
                            error = %error.as_report(),
                            "external coordinator peer connection failed"
                        );
                        continue;
                    }
                    Err(error) => return Err(error.into()),
                };
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

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt as _;
    use tokio::runtime::Handle;
    use tokio::time::timeout;
    use zeromq::prelude::{Socket, SocketRecv, SocketSend};
    use zeromq::{RouterSocket, SocketEvent, XPubSocket, XSubSocket, ZmqMessage};

    use super::ExternalCoordinatorService;
    use crate::client::imp::ClientInner;
    use crate::coordinator::handle::{CoordinatorStateSnapshot, CoordinatorStateUpdate};
    use crate::mock_engine::default_ready_response;
    use crate::protocol::encode_msgpack;
    use crate::test_utils::{read_zmtp_frame, reset_zmtp_stream, zmtp_handshake};
    use crate::transport::{ConnectedEngine, EngineId};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn service_continues_after_peer_connection_error() {
        let mut coordinator = XPubSocket::new();
        let coordinator_address = coordinator.bind("tcp://127.0.0.1:0").await.unwrap().to_string();

        let reset_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let reset_address = format!("tcp://{}", reset_listener.local_addr().unwrap());
        let reset_peer = tokio::task::spawn_blocking(move || {
            let (mut stream, _) = reset_listener.accept().unwrap();
            zmtp_handshake(&mut stream, b"XPUB");
            assert_eq!(read_zmtp_frame(&mut stream), [1]);
            reset_zmtp_stream(stream);
        });

        let mut socket = XSubSocket::new();
        let mut events = socket.monitor();
        socket.connect(&coordinator_address).await.unwrap();
        socket.connect(&reset_address).await.unwrap();
        socket.subscribe("").await.unwrap();
        assert_eq!(
            coordinator.recv().await.unwrap().into_vec()[0].as_ref(),
            &[1]
        );
        reset_peer.await.unwrap();

        let state = Arc::new(parking_lot::Mutex::new(CoordinatorStateSnapshot {
            current_wave: 0,
            engines_running: false,
        }));
        let service = ExternalCoordinatorService::new(state.clone(), socket);
        let mut input = RouterSocket::new();
        input.bind("tcp://127.0.0.1:0").await.unwrap();
        let (input_send, _) = input.split();
        let inner = Arc::new(ClientInner::new(
            input_send,
            Handle::current(),
            "test-model".to_string(),
            0,
            false,
            false,
            0,
            &[ConnectedEngine {
                engine_id: EngineId::from(b"engine-0"),
                ready_response: default_ready_response(),
            }],
        ));
        let service_task = tokio::spawn(service.run(inner.clone()));

        timeout(Duration::from_secs(5), async {
            while !matches!(events.next().await, Some(SocketEvent::Disconnected(_))) {}
        })
        .await
        .expect("peer disconnection event");
        let publish = encode_msgpack(&CoordinatorStateUpdate {
            counts: None,
            wave: 7,
            engines_running: true,
        })
        .unwrap();
        coordinator.send(ZmqMessage::from(publish)).await.unwrap();

        timeout(Duration::from_secs(5), async {
            loop {
                if state.lock().current_wave == 7 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("coordinator state update after peer reset");
        assert!(inner.is_healthy());

        service_task.abort();
        assert!(service_task.await.unwrap_err().is_cancelled());
    }
}
