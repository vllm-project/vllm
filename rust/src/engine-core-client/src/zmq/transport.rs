use std::time::Duration;

use bytes::Bytes;
use thiserror_ext::AsReport;
use tokio::sync::mpsc;
use tokio::time::timeout;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterRecvHalf, RouterSendHalf, RouterSocket, ZmqMessage};

use crate::client::ReadyMessage;
use crate::error::{Error, Result};
use crate::protocol::{EngineCoreOutputs, decode_msgpack};

pub struct ConnectedTransport {
    pub input_send: RouterSendHalf,
    pub input_recv: RouterRecvHalf,
    pub output_socket: PullSocket,
    pub ready_message: Option<ReadyMessage>,
}

pub async fn bind(
    input_address: &str,
    output_address: &str,
) -> Result<(RouterSocket, PullSocket)> {
    let mut input_socket = RouterSocket::new();
    input_socket.bind(input_address).await?;

    let mut output_socket = PullSocket::new();
    output_socket.bind(output_address).await?;

    Ok((input_socket, output_socket))
}

pub async fn connect(
    input_address: &str,
    output_address: &str,
    engine_identity: &[u8],
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    let (input_socket, output_socket) = bind(input_address, output_address).await?;
    connect_bound(input_socket, output_socket, engine_identity, ready_timeout).await
}

pub async fn connect_bound(
    mut input_socket: RouterSocket,
    output_socket: PullSocket,
    engine_identity: &[u8],
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    let ready = timeout(ready_timeout, input_socket.recv())
        .await
        .map_err(|_| Error::ReadyTimeout {
            timeout: ready_timeout,
        })??;

    let ready_message = decode_ready_message(ready, engine_identity)?;

    let (input_send, input_recv) = input_socket.split();

    Ok(ConnectedTransport {
        input_send,
        input_recv,
        output_socket,
        ready_message,
    })
}

fn decode_ready_message(
    message: ZmqMessage,
    expected_identity: &[u8],
) -> Result<Option<ReadyMessage>> {
    if message.len() != 2 {
        return Err(Error::UnexpectedReadyMessage {
            reason: format!("expected 2 frames, got {}", message.len()),
        });
    }

    let frames = message.into_vec();
    let actual_identity = frames[0].to_vec();
    if actual_identity != expected_identity {
        return Err(Error::UnexpectedReadyIdentity {
            expected: expected_identity.to_vec(),
            actual: actual_identity,
        });
    }

    if frames[1].is_empty() {
        return Ok(None);
    }

    let ready_message: ReadyMessage = decode_msgpack(&frames[1])?;
    if ready_message.status.as_deref() != Some("READY") {
        return Err(Error::UnexpectedReadyMessage {
            reason: format!("unexpected status {:?}", ready_message.status),
        });
    }
    Ok(Some(ready_message))
}

pub async fn send_message(
    input_send: &mut RouterSendHalf,
    engine_identity: &[u8],
    request_type: Bytes,
    payload: Vec<u8>,
) -> Result<()> {
    let message = ZmqMessage::try_from(vec![
        Bytes::copy_from_slice(engine_identity),
        request_type,
        Bytes::from(payload),
    ])
    .expect("router messages must contain identity and payload");

    input_send.send(message).await?;
    Ok(())
}

pub fn spawn_output_loop(
    mut output_socket: PullSocket,
    tx: mpsc::Sender<Result<EngineCoreOutputs>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let message = match output_socket.recv().await {
                Ok(message) => message,
                Err(error) => {
                    let _ = tx.send(Err(Error::Transport(error))).await;
                    return;
                }
            };

            let frame_count = message.len();
            if frame_count != 1 {
                let _ = tx
                    .send(Err(Error::UnsupportedAuxFrames { frame_count }))
                    .await;
                return;
            }

            let frame = message.into_vec().into_iter().next().unwrap();
            let decoded = decode_msgpack(frame.as_ref());
            if tx.send(decoded).await.is_err() {
                tracing::warn!("output loop rx dropped, shutting down output loop");
                return;
            }
        }
    })
}

pub fn spawn_input_monitor(
    mut input_recv: RouterRecvHalf,
    engine_identity: Vec<u8>,
    tx: mpsc::Sender<Result<EngineCoreOutputs>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let message = match input_recv.recv().await {
                Ok(message) => message,
                Err(error) => {
                    let _ = tx
                        .send(Err(Error::ControlClosed(error.to_report_string())))
                        .await;
                    tracing::warn!("input monitor rx dropped, shutting down input monitor");
                    return;
                }
            };

            match decode_ready_message(message, &engine_identity) {
                Ok(_) => continue,
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
            }
        }
    })
}
