use std::path::Path;
use std::time::Duration;

use tokio::time::timeout;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, SubSocket, ZmqMessage};

use crate::EngineId;
use crate::error::{Error, Result, bail_unexpected_handshake_message};
use crate::protocol::handshake::{EngineCoreReadyResponse, HandshakeInitMessage, ReadyMessage};
use crate::protocol::{ModelDtype, decode_msgpack, encode_msgpack};

/// Default model length advertised by reusable mock engine helpers.
pub const DEFAULT_MOCK_MAX_MODEL_LEN: u64 = 1024 * 1024;
/// Default KV block count advertised by reusable mock engine helpers.
pub const DEFAULT_MOCK_NUM_GPU_BLOCKS: u64 = 0;

/// Startup behavior for one mock engine joining a frontend.
#[derive(Debug, Clone)]
pub struct MockEngineConfig {
    /// Whether the engine should advertise itself as local to the frontend.
    pub local: bool,
    /// Whether the engine should advertise itself as headless.
    pub headless: bool,
    /// Engine-ready payload reported after INIT, including max model length,
    /// KV block count, and dtype.
    pub ready_response: EngineCoreReadyResponse,
    /// Maximum time to wait for IPC endpoints to appear before connecting.
    pub connect_timeout: Duration,
}

impl Default for MockEngineConfig {
    fn default() -> Self {
        Self {
            local: false,
            headless: true,
            ready_response: default_ready_response(),
            connect_timeout: Duration::from_secs(5),
        }
    }
}

/// Construct the ready response used by the standalone mock engine CLI.
pub fn default_ready_response() -> EngineCoreReadyResponse {
    EngineCoreReadyResponse {
        max_model_len: DEFAULT_MOCK_MAX_MODEL_LEN,
        num_gpu_blocks: DEFAULT_MOCK_NUM_GPU_BLOCKS,
        dp_stats_address: None,
        dtype: ModelDtype::Float32,
        vllm_version: "test-vllm-version".to_string(),
    }
}

/// Coordinator-side sockets used by one mock engine when coordinator mode
/// is enabled.
pub struct MockCoordinatorSockets {
    /// Subscription socket that receives coordinator broadcasts such as
    /// `START_DP_WAVE`.
    pub input_sub: SubSocket,
    /// Push socket used to send coordinator-only `EngineCoreOutputs` back to
    /// the frontend.
    pub output_push: PushSocket,
}

/// One mock engine's connection to one frontend client.
///
/// vLLM launches one engine-client pair per API server process. A remote
/// engine connects to every advertised input/output pair and uses the request's
/// `client_index` to route outputs back to the originating API server.
pub struct MockEngineDataSockets {
    /// Socket used to receive frontend requests.
    pub dealer: DealerSocket,
    /// Socket used to publish normal request outputs back to the frontend.
    pub push: PushSocket,
}

/// Frontend-facing sockets owned by one mock engine.
pub struct MockEngineSockets {
    /// Decoded INIT message sent by the frontend during handshake.
    pub init: HandshakeInitMessage,
    /// Data sockets for all frontend clients in client-index order.
    ///
    /// For Rust frontend this will always be one socket, while for Python frontend
    /// this may be multiple sockets if there are multiple API server processes.
    pub data_sockets: Vec<MockEngineDataSockets>,
    /// Optional coordinator sockets when the client enabled the in-process
    /// coordinator.
    pub coordinator: Option<MockCoordinatorSockets>,
}

/// Build a HELLO or READY handshake status payload.
fn ready_message(status: &str, config: &MockEngineConfig) -> ReadyMessage {
    ReadyMessage {
        status: Some(status.to_string()),
        local: Some(config.local),
        headless: Some(config.headless),
        parallel_config_hash: None,
    }
}

/// Convert an engine id into a ZMQ DEALER identity.
fn peer_identity(engine_id: impl Into<EngineId>) -> Result<PeerIdentity> {
    let engine_id = engine_id.into();
    PeerIdentity::try_from(engine_id.clone()).map_err(|error| Error::UnexpectedHandshakeMessage {
        message: format!(
            "invalid mock engine identity {:?}: {error}",
            engine_id.to_vec()
        ),
    })
}

/// Wait for an IPC endpoint path to appear before attempting to connect.
async fn wait_for_ipc_endpoint(endpoint: &str, connect_timeout: Duration) -> Result<()> {
    let Some(socket_path) = endpoint.strip_prefix("ipc://") else {
        return Ok(());
    };

    timeout(connect_timeout, async {
        while !Path::new(socket_path).exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .map_err(|_| Error::HandshakeTimeout {
        stage: "mock engine IPC endpoint",
        timeout: connect_timeout,
    })
}

/// Encode the engine-ready response sent on input socket registration.
fn ready_response_payload(config: &MockEngineConfig) -> Result<Vec<u8>> {
    encode_msgpack(&config.ready_response)
}

/// Join a frontend-owned handshake endpoint and open mock engine sockets.
pub async fn connect_to_frontend(
    engine_handshake: impl AsRef<str>,
    engine_id: impl Into<EngineId>,
    config: MockEngineConfig,
) -> Result<MockEngineSockets> {
    let engine_handshake = engine_handshake.as_ref();
    wait_for_ipc_endpoint(engine_handshake, config.connect_timeout).await?;

    let peer_identity = peer_identity(engine_id)?;
    let mut options = SocketOptions::default();
    options.peer_identity(peer_identity.clone());
    let mut handshake = DealerSocket::with_options(options);
    handshake.connect(engine_handshake).await?;
    handshake
        .send(ZmqMessage::from(encode_msgpack(&ready_message(
            "HELLO", &config,
        ))?))
        .await?;

    let init_frames = handshake.recv().await?.into_vec();
    if init_frames.len() != 1 {
        bail_unexpected_handshake_message!(
            "expected one INIT frame from frontend, got {}",
            init_frames.len()
        );
    }
    let init: HandshakeInitMessage = decode_msgpack(init_frames[0].as_ref())?;

    if init.addresses.inputs.is_empty() {
        return Err(Error::UnexpectedHandshakeMessage {
            message: "frontend INIT did not include an input address".to_string(),
        });
    }
    if init.addresses.inputs.len() != init.addresses.outputs.len() {
        return Err(Error::UnexpectedHandshakeMessage {
            message: format!(
                "frontend INIT input/output address count mismatch: {} inputs, {} outputs",
                init.addresses.inputs.len(),
                init.addresses.outputs.len()
            ),
        });
    }

    let mut data_sockets = Vec::with_capacity(init.addresses.inputs.len());
    for (input_address, output_address) in
        init.addresses.inputs.iter().zip(init.addresses.outputs.iter())
    {
        wait_for_ipc_endpoint(input_address, config.connect_timeout).await?;
        wait_for_ipc_endpoint(output_address, config.connect_timeout).await?;

        let mut input_options = SocketOptions::default();
        input_options.peer_identity(peer_identity.clone());
        let mut dealer = DealerSocket::with_options(input_options);
        dealer.connect(input_address).await?;
        dealer.send(ZmqMessage::from(ready_response_payload(&config)?)).await?;

        let mut push = PushSocket::new();
        push.connect(output_address).await?;

        data_sockets.push(MockEngineDataSockets { dealer, push });
    }

    let coordinator = match (
        init.addresses.coordinator_input.as_deref(),
        init.addresses.coordinator_output.as_deref(),
    ) {
        (Some(coordinator_input), Some(coordinator_output)) => {
            let mut input_sub = SubSocket::new();
            input_sub.connect(coordinator_input).await?;
            input_sub.subscribe("").await?;

            let mut output_push = PushSocket::new();
            output_push.connect(coordinator_output).await?;

            let ready = input_sub.recv().await?.into_vec();
            if ready.len() != 1 || ready[0].as_ref() != b"READY" {
                bail_unexpected_handshake_message!(
                    "expected coordinator READY marker, got {:?}",
                    ready
                );
            }

            Some(MockCoordinatorSockets {
                input_sub,
                output_push,
            })
        }
        (None, None) => None,
        _ => bail_unexpected_handshake_message!(
            "coordinator handshake addresses must be both present or both absent"
        ),
    };

    handshake
        .send(ZmqMessage::from(encode_msgpack(&ready_message(
            "READY", &config,
        ))?))
        .await?;

    Ok(MockEngineSockets {
        init,
        data_sockets,
        coordinator,
    })
}

/// Join already-bootstrapped frontend input/output sockets directly.
pub async fn connect_to_bootstrapped_frontend(
    input_address: impl AsRef<str>,
    output_address: impl AsRef<str>,
    engine_id: impl Into<EngineId>,
    config: MockEngineConfig,
) -> Result<(DealerSocket, PushSocket)> {
    let input_address = input_address.as_ref();
    let output_address = output_address.as_ref();
    wait_for_ipc_endpoint(input_address, config.connect_timeout).await?;
    wait_for_ipc_endpoint(output_address, config.connect_timeout).await?;

    let peer_identity = peer_identity(engine_id)?;
    let mut input_options = SocketOptions::default();
    input_options.peer_identity(peer_identity);
    let mut dealer = DealerSocket::with_options(input_options);
    dealer.connect(input_address).await?;
    dealer.send(ZmqMessage::from(ready_response_payload(&config)?)).await?;

    let mut push = PushSocket::new();
    push.connect(output_address).await?;

    Ok((dealer, push))
}
