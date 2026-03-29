use std::time::Duration;

use bytes::Bytes;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, XPubSocket, ZmqMessage};

use crate::error::{Error, Result, bail_unexpected_handshake_message};

/// Engine-facing sockets owned by the in-process coordinator.
pub(crate) struct CoordinatorBootstrap {
    pub input_address: String,
    pub output_address: String,
    pub input_socket: XPubSocket,
    pub output_socket: PullSocket,
}

impl CoordinatorBootstrap {
    /// Bind the engine-facing coordinator sockets on the given host.
    pub(crate) async fn bind(local_host: &str) -> Result<Self> {
        let mut input_socket = XPubSocket::new();
        let input_address = input_socket
            .bind(&format!("tcp://{local_host}:0"))
            .await?
            .to_string();

        let mut output_socket = PullSocket::new();
        let output_address = output_socket
            .bind(&format!("tcp://{local_host}:0"))
            .await?
            .to_string();

        Ok(Self {
            input_address,
            output_address,
            input_socket,
            output_socket,
        })
    }

    /// Complete the engine-facing startup gate before engines are allowed to send handshake READY.
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
        let message = tokio::time::timeout(ready_timeout, input_socket.recv())
            .await
            .map_err(|_| Error::HandshakeTimeout {
                stage: "coordinator engine subscriptions",
                timeout: ready_timeout,
            })??;
        if message.len() != 1 {
            bail_unexpected_handshake_message!(
                "expected 1 frame for coordinator subscription, got {}",
                message.len()
            );
        }

        let frame = message
            .into_vec()
            .into_iter()
            .next()
            .expect("single-frame coordinator subscription message");
        if frame.as_ref() != [0x01] {
            bail_unexpected_handshake_message!(
                "expected coordinator subscription frame [0x01], got {:?}",
                frame.as_ref()
            );
        }
        received += 1;
    }

    Ok(())
}

/// Send the coordinator READY marker to all subscribed engines.
async fn send_ready_to_engines(input_socket: &mut XPubSocket) -> Result<()> {
    input_socket
        .send(ZmqMessage::from(Bytes::from_static(b"READY")))
        .await?;
    Ok(())
}
