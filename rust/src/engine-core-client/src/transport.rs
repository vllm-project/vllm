use std::time::Duration;

use bytes::Bytes;
use thiserror_ext::AsReport;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, error, info, trace, warn};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{PullSocket, RouterSendHalf, RouterSocket, ZmqMessage};

use crate::error::{Error, Result};
use crate::protocol::handshake::{HandshakeAddresses, HandshakeInitMessage, ReadyMessage};
use crate::protocol::{EngineCoreOutputs, decode_msgpack, encode_msgpack};

/// Represents the connected transport components after a successful startup handshake, which the
/// client can use for subsequent communication with the engine.
pub struct ConnectedTransport {
    /// The local address of the input socket that the engine connects to for receiving requests.
    pub input_address: String,
    /// The local address of the output socket that the engine connects to for sending responses.
    pub output_address: String,
    /// The identity of the connected engine.
    pub engine_identity: Vec<u8>,

    /// The sending half of the input socket.
    pub input_send: RouterSendHalf,
    /// The output socket for receiving responses from the engine.
    pub output_socket: PullSocket,

    /// The READY message received from the engine during the handshake.
    pub ready_message: ReadyMessage,
}

/// Connect to the engine through the startup handshake protocol, returning [`ConnectedTransport`].
pub async fn connect(
    handshake_address: &str,
    local_host: &str,
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    info!("waiting for engine to connect");

    // 1. Bind handshake socket and local input/output sockets.
    debug!(
        handshake_address,
        local_host,
        ?ready_timeout,
        "waiting for HELLO from engine"
    );
    let mut handshake_socket = RouterSocket::new();
    handshake_socket.bind(handshake_address).await?;

    let (input_address, mut input_socket, output_address, output_socket) =
        bind_local_sockets(local_host).await?;
    debug!(%input_address, %output_address, "bound local transport sockets");

    // 2. Wait for HELLO from engine, extract engine identity, and send INIT with local socket
    //    addresses.
    let hello = timeout(ready_timeout, handshake_socket.recv())
        .await
        .map_err(|_| Error::HandshakeTimeout {
            stage: "HELLO",
            timeout: ready_timeout,
        })??;
    let (engine_identity, _) = decode_handshake_message(hello, None, "HELLO")?;
    debug!(?engine_identity, "received HELLO from engine");

    send_init_message(
        &mut handshake_socket,
        &engine_identity,
        &input_address,
        &output_address,
    )
    .await?;
    debug!(?engine_identity, "sent INIT to engine");

    // 3. Wait for READY from engine and for the engine to connect to the input socket.
    debug!(?engine_identity, "waiting for READY from engine");
    let ready = timeout(ready_timeout, handshake_socket.recv())
        .await
        .map_err(|_| Error::HandshakeTimeout {
            stage: "READY",
            timeout: ready_timeout,
        })??;
    let (_, ready_message) = decode_handshake_message(ready, Some(&engine_identity), "READY")?;
    debug!(
        ?engine_identity,
        ?ready_message,
        "received READY from engine"
    );

    // 4. Wait for the engine to connect to the input socket and register itself.
    wait_for_input_registration(&mut input_socket, &engine_identity, ready_timeout).await?;
    debug!(?engine_identity, "engine input registered");

    info!("engine connected");

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

/// Bind new input and output sockets.
async fn bind_local_sockets(
    local_host: &str,
) -> Result<(String, RouterSocket, String, PullSocket)> {
    let input_address = allocate_tcp_address(local_host).await?;
    let output_address = allocate_tcp_address(local_host).await?;

    let mut input_socket = RouterSocket::new();
    input_socket.bind(&input_address).await?;

    let mut output_socket = PullSocket::new();
    output_socket.bind(&output_address).await?;

    Ok((input_address, input_socket, output_address, output_socket))
}

/// Allocate a port on the local host and return the corresponding TCP address.
async fn allocate_tcp_address(local_host: &str) -> Result<String> {
    let listener = TcpListener::bind((local_host, 0)).await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(format!("tcp://{local_host}:{port}"))
}

/// Decode a handshake message and validate its structure, identity, and status.
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
    if let Some(expected_identity) = expected_identity
        && actual_identity != expected_identity
    {
        return Err(Error::UnexpectedHandshakeIdentity {
            expected: expected_identity.to_vec(),
            actual: actual_identity,
        });
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

/// Send an INIT message to the engine with the local socket addresses for the engine to connect to,
/// using the handshake socket.
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
        parallel_config: Default::default(),
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

/// Receive the input registration message from the engine and validate its identity.
///
/// There will 2 frames: [identity, empty-payload].
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

/// Send an encoded message to the engine through the input socket.
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

    trace!(
        ?engine_identity,
        frame_count = message.len(),
        "sending ZMQ message"
    );
    input_send.send(message).await?;
    Ok(())
}

/// Run the output loop to receive messages from the engine and send them to the provided channel.
pub async fn run_output_loop(
    mut output_socket: PullSocket,
    tx: mpsc::Sender<Result<EngineCoreOutputs>>,
) {
    loop {
        let message = match output_socket.recv().await {
            Ok(message) => message,
            Err(error) => {
                // If we fail to receive a message from the engine, it's likely that the engine has
                // crashed or become unreachable, so we should notify the client and shut down the
                // output loop.
                error!(error = %error.as_report(), "failed to receive output message");
                let _ = tx.send(Err(Error::Transport(error))).await;
                return;
            }
        };

        let frame_count = message.len();
        trace!(frame_count, "received output message");
        if frame_count != 1 {
            error!(
                frame_count,
                "unsupported auxiliary frames in output message"
            );
            let _ = tx
                .send(Err(Error::UnsupportedAuxFrames { frame_count }))
                .await;
            return;
        }

        let frame = message.into_vec().into_iter().next().unwrap();
        let frame_len = frame.len();
        let decoded = match decode_msgpack(frame.as_ref()) {
            Ok(decoded) => {
                trace!(frame_len, outputs = ?decoded, "decoded output message");
                Ok(decoded)
            }
            Err(error) => {
                // If we fail to decode the message from the engine, notify the client but keep the
                // output loop running to continue processing future messages from the engine.
                warn!(frame_len, error = %error.as_report(), "failed to decode output message");
                Err(error)
            }
        };

        if tx.send(decoded).await.is_err() {
            // If we fail to send the decoded message to the client, it's likely that the client has
            // shut down, so we should shut down the output loop as well.
            warn!("output loop rx dropped, shutting down output loop");
            return;
        }
    }
}
