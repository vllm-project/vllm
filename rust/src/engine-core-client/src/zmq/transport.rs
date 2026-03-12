use std::collections::BTreeMap;
use std::net::TcpListener;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::mpsc;
use tokio::time::timeout;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSendHalf, RouterSocket, ZmqMessage};

use crate::error::{Error, Result};
use crate::protocol::handshake::{HandshakeAddresses, HandshakeInitMessage, ReadyMessage};
use crate::protocol::{EngineCoreOutputs, decode_msgpack, encode_msgpack};

pub struct ConnectedTransport {
    pub input_address: String,
    pub output_address: String,
    pub engine_identity: Vec<u8>,
    pub input_send: RouterSendHalf,
    pub output_socket: PullSocket,
    pub ready_message: ReadyMessage,
}

pub async fn connect(
    handshake_address: &str,
    local_host: &str,
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    let mut handshake_socket = RouterSocket::new();
    handshake_socket.bind(handshake_address).await?;

    let (input_address, mut input_socket, output_address, output_socket) =
        bind_local_sockets(local_host).await?;

    let hello = timeout(ready_timeout, handshake_socket.recv())
        .await
        .map_err(|_| Error::HandshakeTimeout {
            stage: "HELLO",
            timeout: ready_timeout,
        })??;
    let (engine_identity, _) = decode_handshake_message(hello, None, "HELLO")?;

    send_init_message(
        &mut handshake_socket,
        &engine_identity,
        &input_address,
        &output_address,
    )
    .await?;

    let ready = timeout(ready_timeout, handshake_socket.recv())
        .await
        .map_err(|_| Error::HandshakeTimeout {
            stage: "READY",
            timeout: ready_timeout,
        })??;
    let (_, ready_message) = decode_handshake_message(ready, Some(&engine_identity), "READY")?;
    wait_for_input_registration(&mut input_socket, &engine_identity, ready_timeout).await?;

    let (input_send, _) = input_socket.split();

    Ok(ConnectedTransport {
        input_address,
        output_address,
        engine_identity,
        input_send,
        output_socket,
        ready_message,
    })
}

async fn bind_local_sockets(
    local_host: &str,
) -> Result<(String, RouterSocket, String, PullSocket)> {
    let input_address = allocate_tcp_address(local_host)?;
    let output_address = allocate_tcp_address(local_host)?;

    let mut input_socket = RouterSocket::new();
    input_socket.bind(&input_address).await?;

    let mut output_socket = PullSocket::new();
    output_socket.bind(&output_address).await?;

    Ok((input_address, input_socket, output_address, output_socket))
}

fn allocate_tcp_address(local_host: &str) -> Result<String> {
    let listener = TcpListener::bind((local_host, 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(format!("tcp://{local_host}:{port}"))
}

fn decode_handshake_message(
    message: ZmqMessage,
    expected_identity: Option<&[u8]>,
    expected_status: &'static str,
) -> Result<(Vec<u8>, ReadyMessage)> {
    if message.len() != 2 {
        return Err(Error::UnexpectedHandshakeMessage {
            reason: format!("expected 2 frames, got {}", message.len()),
        });
    }

    let frames = message.into_vec();
    let actual_identity = frames[0].to_vec();
    if let Some(expected_identity) = expected_identity {
        if actual_identity != expected_identity {
            return Err(Error::UnexpectedHandshakeIdentity {
                expected: expected_identity.to_vec(),
                actual: actual_identity,
            });
        }
    }

    let handshake_message: ReadyMessage = decode_msgpack(&frames[1])?;
    if handshake_message.status.as_deref() != Some(expected_status) {
        return Err(Error::UnexpectedHandshakeMessage {
            reason: format!(
                "expected status {expected_status:?}, got {:?}",
                handshake_message.status
            ),
        });
    }

    Ok((actual_identity, handshake_message))
}

async fn send_init_message(
    handshake_socket: &mut RouterSocket,
    engine_identity: &[u8],
    input_address: &str,
    output_address: &str,
) -> Result<()> {
    let init_message = HandshakeInitMessage {
        addresses: HandshakeAddresses {
            inputs: vec![input_address.to_string()],
            outputs: vec![output_address.to_string()],
            coordinator_input: None,
            coordinator_output: None,
            frontend_stats_publish_address: None,
        },
        parallel_config: BTreeMap::new(),
    };
    let payload = encode_msgpack(&init_message)?;
    let message = ZmqMessage::try_from(vec![
        Bytes::copy_from_slice(engine_identity),
        Bytes::from(payload),
    ])
    .expect("handshake router messages must contain identity and payload");
    handshake_socket.send(message).await?;
    Ok(())
}

async fn wait_for_input_registration(
    input_socket: &mut RouterSocket,
    expected_identity: &[u8],
    ready_timeout: Duration,
) -> Result<()> {
    let registration = timeout(ready_timeout, input_socket.recv())
        .await
        .map_err(|_| Error::InputRegistrationTimeout {
            timeout: ready_timeout,
        })??;

    if registration.len() != 2 {
        return Err(Error::UnexpectedHandshakeMessage {
            reason: format!(
                "expected 2 frames for engine input registration, got {}",
                registration.len()
            ),
        });
    }

    let frames = registration.into_vec();
    let actual_identity = frames[0].to_vec();
    if actual_identity != expected_identity {
        return Err(Error::UnexpectedHandshakeIdentity {
            expected: expected_identity.to_vec(),
            actual: actual_identity,
        });
    }
    if !frames[1].is_empty() {
        return Err(Error::UnexpectedHandshakeMessage {
            reason: "expected empty payload for engine input registration".to_string(),
        });
    }

    Ok(())
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
