use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::ops::Deref;
use std::time::Duration;

use bytes::Bytes;
use thiserror_ext::AsReport;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, error, info, trace, warn};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{PullSocket, RouterSendHalf, RouterSocket, ZmqError, ZmqMessage};

use crate::error::{Error, Result};
use crate::protocol::handshake::{HandshakeAddresses, HandshakeInitMessage, ReadyMessage};
use crate::protocol::{
    EngineCoreOutputs, decode_engine_core_outputs, decode_msgpack, encode_msgpack,
};

/// Dedicated single-frame sentinel emitted by Python `EngineCoreProc` when the engine dies.
pub const ENGINE_CORE_DEAD_SENTINEL: &[u8] = b"ENGINE_CORE_DEAD";

/// Opaque routing identity of one engine on the frontend transport.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EngineId(Bytes);

impl Debug for EngineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display the engine id as a hex string for easier debugging.
        write!(f, "EngineId({})", hex::encode(&self.0))
    }
}

impl EngineId {
    /// Convert the engine id into a ZMQ frame for sending.
    pub fn to_frame(&self) -> Bytes {
        self.0.clone()
    }

    /// Convert the engine id into a ZMQ frame for sending.
    pub fn into_frame(self) -> Bytes {
        self.0
    }
}

impl Deref for EngineId {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl From<Vec<u8>> for EngineId {
    fn from(value: Vec<u8>) -> Self {
        Self(Bytes::from(value))
    }
}

impl<const N: usize> From<&[u8; N]> for EngineId {
    fn from(value: &[u8; N]) -> Self {
        Self(Bytes::copy_from_slice(value))
    }
}

impl TryFrom<EngineId> for PeerIdentity {
    type Error = ZmqError;

    fn try_from(value: EngineId) -> std::result::Result<Self, Self::Error> {
        PeerIdentity::try_from(value.into_frame())
    }
}

/// Per-engine handshake result collected while bootstrapping one shared transport.
#[derive(Clone)]
pub struct ConnectedEngine {
    /// The identity of the connected engine.
    pub engine_id: EngineId,
    /// The READY message received from the engine during the handshake.
    pub ready_message: ReadyMessage,
}

/// Represents the connected shared transport plus all registered engines after a successful
/// multi-engine startup handshake.
pub struct ConnectedTransport {
    /// The local address of the shared input socket that all engines connect to for receiving
    /// requests.
    pub input_address: String,
    /// The local address of the shared output socket that all engines connect to for sending
    /// responses.
    pub output_address: String,
    /// All engines connected through the startup handshake.
    pub engines: Vec<ConnectedEngine>,

    /// The sending half of the shared input socket.
    pub input_send: RouterSendHalf,
    /// The shared output socket for receiving responses from all engines.
    pub output_socket: PullSocket,
}

/// Connect to one or more engines through the startup handshake protocol, returning the shared
/// data-plane transport plus the registered engines.
pub async fn connect(
    handshake_address: &str,
    engine_count: usize,
    local_host: &str,
    local_input_address: Option<&str>,
    local_output_address: Option<&str>,
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    if engine_count == 0 {
        return Err(Error::UnexpectedHandshakeMessage {
            reason: "expected engine_count >= 1".to_string(),
        });
    }

    info!(
        engine_count,
        handshake_address, "waiting for engines to connect"
    );

    // 1. Bind shared local input/output sockets first so every engine receives the same data-plane
    //    addresses during handshake.
    debug!(
        local_host,
        ?ready_timeout,
        engine_count,
        "binding shared transport sockets"
    );
    let (input_address, mut input_socket, output_address, output_socket) =
        bind_local_sockets(local_host, local_input_address, local_output_address).await?;
    debug!(%input_address, %output_address, "bound local transport sockets");

    // 2. Bind the shared handshake socket once. All engines connect to this socket with their own
    //    identities, and startup order does not matter.
    let mut handshake_socket = RouterSocket::new();
    handshake_socket.bind(handshake_address).await?;

    // 3. Receive HELLO/READY from each engine on the shared handshake socket until all expected
    //    engines have completed startup. READY can arrive before later HELLOs, so startup must be
    //    driven by a single mixed-status loop.
    let mut engines = BTreeMap::new();
    let mut ready_pending = BTreeSet::new();
    while engines.len() < engine_count || !ready_pending.is_empty() {
        debug!(
            handshake_address,
            connected = engines.len(),
            ready = engines.len() - ready_pending.len(),
            waiting_for = engine_count,
            "waiting for engine startup message"
        );
        let message = timeout(ready_timeout, handshake_socket.recv())
            .await
            .map_err(|_| Error::HandshakeTimeout {
                stage: "HELLO/READY",
                timeout: ready_timeout,
            })??;
        let (engine_id, handshake_message) = decode_handshake_message(message, None)?;
        match handshake_message.status.as_deref() {
            Some("HELLO") => {
                if engines.len() >= engine_count {
                    return Err(Error::UnexpectedHandshakeMessage {
                        reason: format!(
                            "received HELLO for unexpected extra engine id {engine_id:?}"
                        ),
                    });
                }
                if engines.contains_key(&engine_id) {
                    return Err(Error::UnexpectedHandshakeMessage {
                        reason: format!(
                            "duplicate engine id {engine_id:?} observed during startup handshake"
                        ),
                    });
                }
                debug!(handshake_address, ?engine_id, "received HELLO from engine");

                send_init_message(
                    &mut handshake_socket,
                    &engine_id,
                    &input_address,
                    &output_address,
                )
                .await?;
                debug!(handshake_address, ?engine_id, "sent INIT to engine");

                engines.insert(
                    engine_id.clone(),
                    ConnectedEngine {
                        engine_id: engine_id.clone(),
                        // Haven't received READY yet, use a placeholder for now.
                        ready_message: ReadyMessage::default(),
                    },
                );
                ready_pending.insert(engine_id);
            }
            Some("READY") => {
                if !ready_pending.remove(&engine_id) {
                    return Err(Error::UnexpectedHandshakeMessage {
                        reason: format!(
                            "received READY for unexpected or duplicate engine id {engine_id:?}"
                        ),
                    });
                }
                debug!(
                    handshake_address,
                    ?engine_id,
                    ?handshake_message,
                    "received READY from engine"
                );
                engines
                    .get_mut(&engine_id)
                    .expect("READY must only be accepted for a previously HELLO'd engine")
                    .ready_message = handshake_message;
            }
            other => {
                return Err(Error::UnexpectedHandshakeMessage {
                    reason: format!("unexpected handshake status {other:?}"),
                });
            }
        }
    }

    // 4. Wait for every engine to connect to the shared input socket and register itself.
    let engines: Vec<_> = engines.into_values().collect();

    wait_for_input_registrations(&mut input_socket, &engines, ready_timeout).await?;
    debug!(
        engine_count = engines.len(),
        "all engines registered on shared input socket"
    );

    info!(engine_count = engines.len(), "engines connected");

    let (input_send, _) = input_socket.split();

    Ok(ConnectedTransport {
        input_address,
        output_address,
        input_send,
        output_socket,
        engines,
    })
}

/// Bind new input and output sockets.
async fn bind_local_sockets(
    local_host: &str,
    local_input_address: Option<&str>,
    local_output_address: Option<&str>,
) -> Result<(String, RouterSocket, String, PullSocket)> {
    let mut input_socket = RouterSocket::new();
    let input_bind_address = local_input_address
        .map(str::to_owned)
        .unwrap_or_else(|| format!("tcp://{local_host}:0"));
    let input_address = input_socket.bind(&input_bind_address).await?.to_string();

    let mut output_socket = PullSocket::new();
    let output_bind_address = local_output_address
        .map(str::to_owned)
        .unwrap_or_else(|| format!("tcp://{local_host}:0"));
    let output_address = output_socket.bind(&output_bind_address).await?.to_string();

    Ok((input_address, input_socket, output_address, output_socket))
}

/// Decode a handshake message and validate its structure and identity.
fn decode_handshake_message(
    message: ZmqMessage,
    expected_id: Option<&EngineId>,
) -> Result<(EngineId, ReadyMessage)> {
    if message.len() != 2 {
        return Err(Error::UnexpectedHandshakeMessage {
            reason: format!("expected 2 frames, got {}", message.len()),
        });
    }

    let frames = message.into_vec();
    let actual_id = EngineId(frames[0].clone());
    if let Some(expected_id) = expected_id
        && actual_id != *expected_id
    {
        return Err(Error::UnexpectedHandshakeIdentity {
            expected: expected_id.to_vec(),
            actual: actual_id.to_vec(),
        });
    }

    let handshake_message: ReadyMessage = decode_msgpack(&frames[1])?;
    Ok((actual_id, handshake_message))
}

/// Send an INIT message to the engine with the local socket addresses for the engine to connect to,
/// using the handshake socket.
async fn send_init_message(
    handshake_socket: &mut RouterSocket,
    engine_id: &EngineId,
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
    let message = ZmqMessage::try_from(vec![engine_id.to_frame(), Bytes::from(payload)])
        .expect("handshake router messages must contain identity and payload");
    handshake_socket.send(message).await?;
    Ok(())
}

/// Receive the input registration message from each engine and validate its identity.
///
/// Each registration contains 2 frames: `[identity, empty-payload]`.
async fn wait_for_input_registrations(
    input_socket: &mut RouterSocket,
    expected_engines: &[ConnectedEngine],
    ready_timeout: Duration,
) -> Result<()> {
    let mut pending = expected_engines
        .iter()
        .map(|e| e.engine_id.clone())
        .collect::<BTreeSet<_>>();

    while !pending.is_empty() {
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
        let actual_id = EngineId(frames[0].clone());
        if !pending.remove(&actual_id) {
            return Err(Error::UnexpectedHandshakeMessage {
                reason: format!(
                    "received input registration for unexpected engine id {actual_id:?}"
                ),
            });
        }
        if !frames[1].is_empty() {
            return Err(Error::UnexpectedHandshakeMessage {
                reason: "expected empty payload for engine input registration".to_string(),
            });
        }
    }

    Ok(())
}

/// Send an encoded message to the engine through the input socket.
pub async fn send_message(
    input_send: &mut RouterSendHalf,
    engine_id: &EngineId,
    request_type: Bytes,
    payload: Vec<u8>,
) -> Result<()> {
    let message = ZmqMessage::try_from(vec![
        engine_id.to_frame(),
        request_type,
        Bytes::from(payload),
    ])
    .expect("router messages must contain identity and payload");

    trace!(
        ?engine_id,
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
        let frames = message.into_vec();
        let frame = frames
            .first()
            .expect("output message must have at least one frame");
        let frame_len = frame.len();
        if frame.as_ref() == ENGINE_CORE_DEAD_SENTINEL {
            warn!("received ENGINE_CORE_DEAD sentinel from engine");
            let _ = tx.send(Err(Error::EngineCoreDead)).await;
            return;
        }
        let decoded = match decode_engine_core_outputs(&frames) {
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

#[cfg(test)]
mod tests {
    use super::bind_local_sockets;

    #[tokio::test]
    async fn bind_local_sockets_resolves_zero_port_bindings() {
        let (input_address, _input_socket, output_address, _output_socket) =
            bind_local_sockets("127.0.0.1", None, None)
                .await
                .expect("bind local sockets");

        assert!(input_address.starts_with("tcp://127.0.0.1:"));
        assert!(output_address.starts_with("tcp://127.0.0.1:"));
        assert_ne!(input_address, output_address);
    }
}
