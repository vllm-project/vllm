// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use bytes::Bytes;
use tempfile::TempDir;
use tokio::sync::oneshot;
use zeromq::{DealerSocket, PushSocket};

use crate::EngineId;
pub use crate::mock_engine::{MockCoordinatorSockets, MockEngineSockets};
use crate::mock_engine::{
    MockEngineConfig, MockEngineDataSockets, connect_to_bootstrapped_frontend, connect_to_frontend,
    default_ready_response,
};
use crate::protocol::dtype::{Endianness, TensorDtype};
use crate::protocol::handshake::{EngineCoreReadyResponse, HandshakeInitMessage};
use crate::protocol::request::EngineCoreRequest;

/// Decode the `[request type, payload, aux frames...]` of an add request,
/// resolving its prompt token ids if they were sent in an auxiliary frame.
pub fn decode_add_request(frames: &[Bytes]) -> EngineCoreRequest {
    let mut request: EngineCoreRequest =
        rmp_serde::from_slice(&frames[1]).expect("decode add request");
    if let Some(ids) = &mut request.prompt_token_ids {
        ids.resolve_aux_frame(&frames[1..]).expect("resolve prompt token ids");
    }
    request
}

/// The prompt token ids of a request, which must be inline or resolved by
/// [`decode_add_request`].
pub fn decode_prompt_token_ids(request: &EngineCoreRequest) -> Option<Vec<u32>> {
    let ids = request.prompt_token_ids.as_ref()?;
    assert!(matches!(
        ids.dtype.scalar,
        TensorDtype::U32 | TensorDtype::I32
    ));
    let bytes = ids.data.as_raw_view().expect("prompt token ids are inline");
    let decode = match ids.dtype.endianness {
        Endianness::Little => u32::from_le_bytes,
        Endianness::Big => u32::from_be_bytes,
        Endianness::Native => u32::from_ne_bytes,
    };
    Some(bytes.chunks_exact(4).map(|c| decode(c.try_into().unwrap())).collect())
}

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

fn test_mock_engine_config() -> MockEngineConfig {
    MockEngineConfig {
        local: true,
        headless: true,
        ready_response: default_ready_response(),
        ..Default::default()
    }
}

fn test_mock_engine_config_with_ready(ready_response: EngineCoreReadyResponse) -> MockEngineConfig {
    MockEngineConfig {
        local: true,
        headless: true,
        ready_response,
        ..Default::default()
    }
}

/// Complete the engine-core handshake and connect mock input/output sockets
/// plus optional coordinator sockets.
pub async fn setup_mock_engine_sockets(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
) -> MockEngineSockets {
    connect_to_frontend(engine_handshake, engine_id, test_mock_engine_config())
        .await
        .expect("connect mock engine")
}

/// Connect one mock engine directly to already-bootstrapped frontend
/// input/output sockets.
pub async fn setup_bootstrapped_mock_engine(
    input_address: String,
    output_address: String,
    engine_id: impl Into<EngineId>,
) -> (DealerSocket, PushSocket) {
    connect_to_bootstrapped_frontend(
        input_address,
        output_address,
        engine_id,
        test_mock_engine_config(),
    )
    .await
    .expect("connect bootstrapped mock engine")
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
    let MockEngineSockets {
        init, data_sockets, ..
    } = setup_mock_engine_sockets(engine_handshake, engine_id).await;
    let MockEngineDataSockets { dealer, push } =
        data_sockets.into_iter().next().expect("mock engine data socket");
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

/// Like [`setup_mock_engine`] but uses a custom ready response for the
/// handshake, allowing tests to control `world_size`, `data_parallel_size`,
/// etc.
async fn setup_mock_engine_with_ready(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
    ready_response: EngineCoreReadyResponse,
) -> (DealerSocket, PushSocket) {
    let config = test_mock_engine_config_with_ready(ready_response);
    let MockEngineSockets { data_sockets, .. } =
        connect_to_frontend(engine_handshake, engine_id, config)
            .await
            .expect("connect mock engine with custom ready response");
    let MockEngineDataSockets { dealer, push } =
        data_sockets.into_iter().next().expect("mock engine data socket");
    (dealer, push)
}

/// Like [`spawn_mock_engine_task`] but uses a custom ready response for the
/// handshake, allowing tests to set `world_size` and `data_parallel_size` to
/// non-default values.
pub fn spawn_mock_engine_task_with_ready<F>(
    engine_handshake: String,
    engine_id: impl Into<EngineId>,
    ready_response: EngineCoreReadyResponse,
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
        let (mut dealer, mut push) =
            setup_mock_engine_with_ready(engine_handshake, engine_id, ready_response).await;
        run(&mut dealer, &mut push).await;
        let _ = shutdown_rx.await;
    });
    (shutdown_tx, engine_task)
}
