use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::Duration;

use tempfile::TempDir;
use tokio::sync::oneshot;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, SubSocket, ZmqMessage};

use crate::EngineId;
use crate::protocol::handshake::{EngineCoreReadyResponse, HandshakeInitMessage, ReadyMessage};

/// Per-test IPC endpoint namespace backed by a unique temporary directory.
///
/// Using one directory per test avoids endpoint collisions without requiring
/// ad-hoc unique-name generation at each call site.
#[derive(Debug)]
pub struct IpcNamespace {
    dir: TempDir,
}

impl IpcNamespace {
    /// Create a fresh namespace for one test case.
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            dir: TempDir::new()?,
        })
    }

    /// Build one `ipc://...` endpoint under this namespace.
    pub fn endpoint(&self, name: impl AsRef<Path>) -> String {
        let path = self.dir.path().join(name);
        format!("ipc://{}", path.to_string_lossy())
    }

    /// Endpoint used for the initial READY/HELLO handshake.
    pub fn handshake_endpoint(&self) -> String {
        self.endpoint("handshake.sock")
    }

    /// Endpoint used for engine-core request traffic.
    pub fn input_endpoint(&self) -> String {
        self.endpoint("input.sock")
    }

    /// Endpoint used for engine-core output traffic.
    pub fn output_endpoint(&self) -> String {
        self.endpoint("output.sock")
    }
}

/// Construct a standard local READY message used by mock engines in tests.
fn ready_message(status: &str) -> ReadyMessage {
    ReadyMessage {
        status: Some(status.to_string()),
        local: Some(true),
        headless: Some(true),
        parallel_config_hash: None,
    }
}

/// Construct a default ready response payload for mock engine input
/// registration.
fn ready_response_payload() -> Vec<u8> {
    rmp_serde::to_vec_named(&EngineCoreReadyResponse {
        max_model_len: 4096,
        num_gpu_blocks: 0,
        dp_stats_address: None,
    })
    .expect("encode ready response payload")
}

/// Coordinator-side sockets connected by one mock engine when coordinator mode
/// is enabled.
pub struct MockCoordinatorConnections {
    /// Subscription socket that receives coordinator broadcasts such as
    /// `START_DP_WAVE`.
    pub input_sub: SubSocket,
    /// Push socket used to send coordinator-only `EngineCoreOutputs` back to
    /// the frontend.
    pub output_push: PushSocket,
}

/// Fully connected mock engine transport state used by tests.
pub struct MockEngineConnections {
    /// Decoded INIT message sent by the frontend during handshake.
    pub init: HandshakeInitMessage,
    /// Socket used to receive frontend requests.
    pub dealer: DealerSocket,
    /// Socket used to publish normal request outputs back to the frontend.
    pub push: PushSocket,
    /// Optional coordinator sockets when the client enabled the in-process
    /// coordinator.
    pub coordinator: Option<MockCoordinatorConnections>,
}

/// Complete the engine-core handshake and connect mock input/output sockets
/// plus optional coordinator sockets.
pub async fn setup_mock_engine_connections(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
) -> MockEngineConnections {
    // Wait for the client to bind the handshake socket before connecting.
    // A fixed sleep is racy under CI load; instead poll for the socket file.
    let socket_path = engine_handshake
        .strip_prefix("ipc://")
        .expect("handshake address must be ipc://");
    for _ in 0..100 {
        if Path::new(socket_path).exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let peer_identity = PeerIdentity::try_from(engine_id.into()).expect("peer id");

    let mut options = SocketOptions::default();
    options.peer_identity(peer_identity.clone());
    let mut handshake = DealerSocket::with_options(options);
    handshake
        .connect(&engine_handshake)
        .await
        .expect("connect mock engine handshake socket");
    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("HELLO")).expect("encode HELLO ready message"),
        ))
        .await
        .expect("send HELLO ready message");

    let init_frames = handshake.recv().await.expect("receive handshake init message").into_vec();
    assert_eq!(init_frames.len(), 1);
    let init: HandshakeInitMessage =
        rmp_serde::from_slice(init_frames[0].as_ref()).expect("decode handshake init message");

    let mut input_options = SocketOptions::default();
    input_options.peer_identity(peer_identity);
    let mut dealer = DealerSocket::with_options(input_options);
    dealer
        .connect(&init.addresses.inputs[0])
        .await
        .expect("connect mock engine input socket");
    dealer
        .send(ZmqMessage::from(ready_response_payload()))
        .await
        .expect("send mock engine input ready frame");

    let mut push = PushSocket::new();
    push.connect(&init.addresses.outputs[0])
        .await
        .expect("connect mock engine output socket");

    let coordinator = match (
        init.addresses.coordinator_input.as_deref(),
        init.addresses.coordinator_output.as_deref(),
    ) {
        (Some(coordinator_input), Some(coordinator_output)) => {
            let mut input_sub = SubSocket::new();
            input_sub
                .connect(coordinator_input)
                .await
                .expect("connect mock engine coordinator input socket");
            input_sub
                .subscribe("")
                .await
                .expect("subscribe mock engine coordinator input socket");

            let mut output_push = PushSocket::new();
            output_push
                .connect(coordinator_output)
                .await
                .expect("connect mock engine coordinator output socket");

            let ready =
                input_sub.recv().await.expect("receive coordinator READY marker").into_vec();
            assert_eq!(ready.len(), 1);
            assert_eq!(ready[0].as_ref(), b"READY");

            Some(MockCoordinatorConnections {
                input_sub,
                output_push,
            })
        }
        (None, None) => None,
        _ => panic!("coordinator handshake addresses must be both present or both absent"),
    };

    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("READY")).expect("encode READY ready message"),
        ))
        .await
        .expect("send READY ready message");

    MockEngineConnections {
        init,
        dealer,
        push,
        coordinator,
    }
}

/// Connect one mock engine directly to already-bootstrapped frontend
/// input/output sockets.
pub async fn setup_bootstrapped_mock_engine(
    input_address: String,
    output_address: String,
    engine_id: impl Into<EngineId>,
) -> (DealerSocket, PushSocket) {
    for endpoint in [&input_address, &output_address] {
        if let Some(socket_path) = endpoint.strip_prefix("ipc://") {
            for _ in 0..100 {
                if Path::new(socket_path).exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }

    let peer_identity = PeerIdentity::try_from(engine_id.into()).expect("peer id");
    let mut input_options = SocketOptions::default();
    input_options.peer_identity(peer_identity);
    let mut dealer = DealerSocket::with_options(input_options);
    dealer.connect(&input_address).await.expect("connect mock engine input socket");
    dealer
        .send(ZmqMessage::from(ready_response_payload()))
        .await
        .expect("send mock engine input ready frame");

    let mut push = PushSocket::new();
    push.connect(&output_address).await.expect("connect mock engine output socket");

    (dealer, push)
}

/// Complete the engine-core handshake and connect mock input/output sockets.
///
/// This returns the decoded handshake init message plus the `DealerSocket` used
/// to receive client requests and the `PushSocket` used to send engine outputs
/// back to the client.
pub async fn setup_mock_engine_with_init(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
) -> (HandshakeInitMessage, DealerSocket, PushSocket) {
    let MockEngineConnections {
        init, dealer, push, ..
    } = setup_mock_engine_connections(engine_handshake, engine_id).await;
    (init, dealer, push)
}

/// Complete the engine-core handshake and connect mock input/output sockets.
///
/// This returns the `DealerSocket` used to receive client requests and the
/// `PushSocket` used to send engine outputs back to the client.
pub async fn setup_mock_engine(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
) -> (DealerSocket, PushSocket) {
    let (_, dealer, push) = setup_mock_engine_with_init(engine_handshake, engine_id).await;
    (dealer, push)
}

/// Spawn a mock engine task and keep its sockets alive until the returned
/// shutdown sender is triggered by the test.
///
/// The script borrows the connected sockets mutably while it runs. After the
/// script completes, this helper keeps the sockets alive until the test
/// explicitly signals shutdown.
pub fn spawn_mock_engine_task<F>(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
    run: F,
) -> (oneshot::Sender<()>, tokio::task::JoinHandle<()>)
where
    F: for<'a> FnOnce(
            &'a mut DealerSocket,
            &'a mut PushSocket,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>
        + Send
        + 'static,
{
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let engine_id = engine_id.into();
    let engine_task = tokio::spawn(async move {
        let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_id).await;
        run(&mut dealer, &mut push).await;
        let _ = shutdown_rx.await;
    });
    (shutdown_tx, engine_task)
}
