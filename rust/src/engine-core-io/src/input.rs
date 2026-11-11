// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The input socket thread (`EngineCoreProc.process_input_sockets`).
//!
//! The thread connects the input sockets, announces the engine on each with
//! the ready response, joins the DP coordinator if there is one, then hands
//! every message it receives to the handler, in order, on this thread.

use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::thread;

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::{DealerSocket, SocketOptions, SubSocket, ZmqMessage};

use crate::output::ErrorHandler;
use crate::stopped::{Stopped, StoppedGuard};

/// The DP coordinator's marker that it is ready, and later that new engines
/// started; the engine ignores the latter.
const READY: &[u8] = b"READY";

/// Where and how the engine core connects its input sockets.
#[derive(Clone, Debug)]
pub struct InputConfig {
    /// One input socket per frontend client.
    pub input_addresses: Vec<String>,
    /// The DP coordinator's input socket, if there is one.
    pub coordinator_address: Option<String>,
    /// The engine's identity on the frontends' ROUTER sockets.
    pub identity: Bytes,
    /// The encoded `EngineCoreReadyResponse`, sent first on each input
    /// socket: the frontend can send requests only after it.
    pub ready_payload: Bytes,
}

/// Handles one received message, `[request type, payload, aux frames...]`.
/// An error stops the thread.
pub type MessageHandler = Box<dyn FnMut(Vec<Bytes>) -> Result<(), String> + Send>;

/// Handle to the input socket thread.
pub struct InputReader {
    stopped: Arc<Stopped>,
}

impl InputReader {
    /// Start the thread; returns once the input sockets are connected and the
    /// ready response sent. `on_ready` runs once the DP coordinator is ready
    /// too, before the first message is handled.
    pub fn start(
        config: InputConfig,
        on_ready: Box<dyn FnOnce() + Send>,
        mut on_message: MessageHandler,
        on_error: ErrorHandler,
    ) -> Result<Self, String> {
        let (connected_tx, connected_rx) = std_mpsc::channel();
        let stopped = Arc::new(Stopped::default());
        let guard = StoppedGuard(stopped.clone());
        thread::Builder::new()
            .name("engine-core-input".to_string())
            .spawn(move || {
                let _guard = guard;
                let runtime =
                    match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            let _ = connected_tx.send(Err(error.to_string()));
                            return;
                        }
                    };
                runtime.block_on(async move {
                    let dealers = match connect_inputs(&config).await {
                        Ok(dealers) => {
                            let _ = connected_tx.send(Ok(()));
                            dealers
                        }
                        Err(error) => {
                            let _ = connected_tx.send(Err(error));
                            return;
                        }
                    };
                    let result = async {
                        let coordinator = match &config.coordinator_address {
                            Some(address) => Some(connect_coordinator(address).await?),
                            None => None,
                        };
                        on_ready();
                        let mut messages = merge(dealers, coordinator);
                        while let Some(message) = messages.next().await {
                            let frames = message?.into_vec();
                            if frames.first().is_some_and(|f| f.as_ref() == READY) {
                                continue;
                            }
                            on_message(frames)?;
                        }
                        Ok::<_, String>(())
                    }
                    .await;
                    if let Err(error) = result {
                        on_error(error);
                    }
                });
            })
            .map_err(|error| error.to_string())?;
        match connected_rx.recv() {
            Ok(Ok(())) => Ok(Self { stopped }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err("the input thread stopped while connecting".to_string()),
        }
    }

    /// Whether the thread is still running.
    pub fn is_alive(&self) -> bool {
        !self.stopped.get()
    }
}

async fn connect_inputs(config: &InputConfig) -> Result<Vec<DealerSocket>, String> {
    let identity = zeromq::util::PeerIdentity::try_from(config.identity.to_vec())
        .map_err(|error| format!("invalid engine identity: {error}"))?;
    let mut dealers = Vec::with_capacity(config.input_addresses.len());
    for address in &config.input_addresses {
        let mut options = SocketOptions::default();
        options.peer_identity(identity.clone());
        let mut dealer = DealerSocket::with_options(options);
        dealer
            .connect(address)
            .await
            .map_err(|error| format!("failed to connect input socket {address}: {error}"))?;
        // Required before the frontend's ROUTER socket can send to us.
        dealer
            .send(ZmqMessage::from(config.ready_payload.clone()))
            .await
            .map_err(|error| error.to_string())?;
        dealers.push(dealer);
    }
    Ok(dealers)
}

/// Subscribe to the DP coordinator and wait for its READY.
async fn connect_coordinator(address: &str) -> Result<SubSocket, String> {
    let mut socket = SubSocket::new();
    socket
        .connect(address)
        .await
        .map_err(|error| format!("failed to connect coordinator socket {address}: {error}"))?;
    socket.subscribe("").await.map_err(|error| error.to_string())?;
    let ready = socket.recv().await.map_err(|error| error.to_string())?.into_vec();
    if ready.len() != 1 || ready[0].as_ref() != READY {
        return Err(format!("expected the coordinator's READY, got {ready:?}"));
    }
    Ok(socket)
}

type Received = Result<ZmqMessage, String>;

/// The messages of every input socket, as they arrive.
fn merge(
    dealers: Vec<DealerSocket>,
    coordinator: Option<SubSocket>,
) -> stream::SelectAll<BoxStream<'static, Received>> {
    fn messages<S: SocketRecv + Send + 'static>(socket: S) -> BoxStream<'static, Received> {
        stream::unfold(socket, |mut socket| async move {
            let message = socket.recv().await.map_err(|error| error.to_string());
            Some((message, socket))
        })
        .boxed()
    }
    let mut streams: Vec<_> = dealers.into_iter().map(messages).collect();
    streams.extend(coordinator.map(messages));
    stream::select_all(streams)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use vllm_engine_core_client::test_utils::IpcNamespace;
    use zeromq::RouterSocket;

    use super::*;

    #[tokio::test]
    async fn announces_the_engine_then_hands_over_messages_in_order() {
        let ipc = IpcNamespace::new().unwrap();
        let address = ipc.endpoint("input");
        let mut router = RouterSocket::new();
        router.bind(&address).await.unwrap();

        // Async, so that waiting lets the ROUTER's writer task run.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let ready = Arc::new(Mutex::new(false));
        let ready_set = ready.clone();
        let reader = tokio::task::spawn_blocking(move || {
            InputReader::start(
                InputConfig {
                    input_addresses: vec![address],
                    coordinator_address: None,
                    identity: Bytes::from_static(&[1, 0]),
                    ready_payload: Bytes::from_static(b"ready-response"),
                },
                Box::new(move || *ready_set.lock().unwrap() = true),
                Box::new(move |frames| tx.send(frames).map_err(|e| e.to_string())),
                Box::new(|error| panic!("{error}")),
            )
            .unwrap()
        })
        .await
        .unwrap();

        // The ROUTER sees `[identity, payload]`.
        let hello = router.recv().await.unwrap().into_vec();
        assert_eq!(
            hello,
            [
                Bytes::from_static(&[1, 0]),
                Bytes::from_static(b"ready-response")
            ]
        );
        for payload in [&b"first"[..], b"second"] {
            let message = ZmqMessage::try_from(vec![
                Bytes::from_static(&[1, 0]),
                Bytes::from_static(b"\x00"),
                Bytes::copy_from_slice(payload),
            ])
            .unwrap();
            router.send(message).await.unwrap();
        }
        for payload in [&b"first"[..], b"second"] {
            let frames =
                tokio::time::timeout(Duration::from_secs(5), rx.recv()).await.unwrap().unwrap();
            assert_eq!(
                frames,
                [Bytes::from_static(b"\x00"), Bytes::copy_from_slice(payload)]
            );
        }
        assert!(*ready.lock().unwrap());
        assert!(reader.is_alive());
    }
}
