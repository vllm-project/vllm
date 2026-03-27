use std::convert::TryFrom;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::time::Duration;

use tempfile::TempDir;
use tokio::sync::oneshot;
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{DealerSocket, PushSocket, SocketOptions, ZmqMessage};

use crate::protocol::handshake::{HandshakeInitMessage, ReadyMessage};

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
        num_gpu_blocks: None,
        dp_stats_address: None,
        parallel_config_hash: None,
    }
}

/// Complete the engine-core handshake and connect mock input/output sockets.
///
/// This returns the decoded handshake init message plus the `DealerSocket` used to receive client
/// requests and the `PushSocket` used to send engine outputs back to the client.
pub async fn setup_mock_engine_with_init(
    engine_handshake: String,
    engine_identity: Vec<u8>,
) -> (HandshakeInitMessage, DealerSocket, PushSocket) {
    tokio::time::sleep(Duration::from_millis(200)).await;

    let mut options = SocketOptions::default();
    options.peer_identity(
        PeerIdentity::try_from(engine_identity.clone()).expect("encode engine peer identity"),
    );
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

    let init_frames = handshake
        .recv()
        .await
        .expect("receive handshake init message")
        .into_vec();
    assert_eq!(init_frames.len(), 1);
    let init: HandshakeInitMessage =
        rmp_serde::from_slice(init_frames[0].as_ref()).expect("decode handshake init message");

    let mut input_options = SocketOptions::default();
    input_options.peer_identity(
        PeerIdentity::try_from(engine_identity).expect("encode input peer identity"),
    );
    let mut dealer = DealerSocket::with_options(input_options);
    dealer
        .connect(&init.addresses.inputs[0])
        .await
        .expect("connect mock engine input socket");
    dealer
        .send(ZmqMessage::from(Vec::<u8>::new()))
        .await
        .expect("send mock engine input ready frame");

    let mut push = PushSocket::new();
    push.connect(&init.addresses.outputs[0])
        .await
        .expect("connect mock engine output socket");

    handshake
        .send(ZmqMessage::from(
            rmp_serde::to_vec_named(&ready_message("READY")).expect("encode READY ready message"),
        ))
        .await
        .expect("send READY ready message");

    (init, dealer, push)
}

/// Complete the engine-core handshake and connect mock input/output sockets.
///
/// This returns the `DealerSocket` used to receive client requests and the
/// `PushSocket` used to send engine outputs back to the client.
pub async fn setup_mock_engine(
    engine_handshake: String,
    engine_identity: Vec<u8>,
) -> (DealerSocket, PushSocket) {
    let (_, dealer, push) = setup_mock_engine_with_init(engine_handshake, engine_identity).await;
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
    engine_identity: Vec<u8>,
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
    let engine_task = tokio::spawn(async move {
        let (mut dealer, mut push) = setup_mock_engine(engine_handshake, engine_identity).await;
        run(&mut dealer, &mut push).await;
        let _ = shutdown_rx.await;
    });
    (shutdown_tx, engine_task)
}
