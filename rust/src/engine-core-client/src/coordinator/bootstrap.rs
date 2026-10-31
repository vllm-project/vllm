// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use bytes::Bytes;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, XPubSocket, ZmqMessage};

use crate::error::{Error, Result, bail_unexpected_handshake_message};

/// Sockets owned by the in-process coordinator: the engine-facing broadcast
/// and control sockets, and the frontend-facing state publish socket.
pub(crate) struct CoordinatorBootstrap {
    pub input_address: String,
    pub output_address: String,
    /// Sent to engines in INIT, which report it back to every frontend as
    /// READY `dp_stats_address`.
    pub frontend_publish_address: String,
    pub input_socket: XPubSocket,
    pub output_socket: PullSocket,
    pub frontend_publish_socket: XPubSocket,
}

impl CoordinatorBootstrap {
    /// Bind the coordinator sockets on the given host.
    pub(crate) async fn bind(local_host: &str) -> Result<Self> {
        let mut input_socket = XPubSocket::new();
        let input_address = input_socket.bind(&format!("tcp://{local_host}:0")).await?.to_string();

        let mut output_socket = PullSocket::new();
        let output_address =
            output_socket.bind(&format!("tcp://{local_host}:0")).await?.to_string();

        let mut frontend_publish_socket = XPubSocket::new();
        let frontend_publish_address = frontend_publish_socket
            .bind(&format!("tcp://{local_host}:0"))
            .await?
            .to_string();

        Ok(Self {
            input_address,
            output_address,
            frontend_publish_address,
            input_socket,
            output_socket,
            frontend_publish_socket,
        })
    }

    /// Complete the engine-facing startup gate before engines are allowed to
    /// send handshake READY.
    pub(crate) async fn wait_for_startup_gate(
        &mut self,
        engine_count: usize,
        ready_timeout: Duration,
    ) -> Result<()> {
        wait_for_engine_subscriptions(&mut self.input_socket, engine_count, ready_timeout).await?;
        send_ready_to_engines(&mut self.input_socket).await?;
        Ok(())
    }
}

/// Wait until all engines subscribe to the coordinator broadcast socket.
async fn wait_for_engine_subscriptions(
    input_socket: &mut XPubSocket,
    engine_count: usize,
    ready_timeout: Duration,
) -> Result<()> {
    let mut received = 0;
    while received < engine_count {
        let message =
            tokio::time::timeout(ready_timeout, input_socket.recv()).await.map_err(|_| {
                Error::HandshakeTimeout {
                    stage: "coordinator engine subscriptions",
                    timeout: ready_timeout,
                }
            })??;
        if !is_subscribe_all(&message) {
            bail_unexpected_handshake_message!(
                "expected coordinator subscription [0x01], got {message:?}"
            );
        }
        received += 1;
    }

    Ok(())
}

/// Whether `message` is the single-frame subscribe-all notification (`[0x01]`)
/// that an XSUB peer sends whenever it connects or reconnects to a coordinator
/// XPUB socket.
pub(super) fn is_subscribe_all(message: &ZmqMessage) -> bool {
    message.len() == 1 && message.get(0).is_some_and(|frame| frame.as_ref() == [0x01])
}

/// Send the coordinator READY marker to all subscribed engines.
async fn send_ready_to_engines(input_socket: &mut XPubSocket) -> Result<()> {
    input_socket.send(ZmqMessage::from(Bytes::from_static(b"READY"))).await?;
    Ok(())
}
