// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The output socket thread (`EngineCoreProc.process_output_sockets`).
//!
//! Request outputs the scheduler builds natively are encoded here, off the
//! GIL. Everything else the engine sends (utility results, DP coordinator
//! messages, error outputs, ...) is encoded by Python and handed over as
//! frames, through the same channel so that the order of messages is kept.

use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::mpsc;
use vllm_engine_core_client::ENGINE_CORE_DEAD_SENTINEL;
use vllm_engine_core_client::protocol::output::{
    EngineCoreOutputs, RequestBatchOutputs, encode_engine_core_outputs,
};
use zeromq::prelude::{Socket, SocketSend};
use zeromq::{PushSocket, ZmqMessage};

/// Where the engine core sends its output sockets.
#[derive(Clone, Debug)]
pub struct OutputConfig {
    /// One output socket per frontend client, by `client_index`.
    pub client_addresses: Vec<String>,
    /// The DP coordinator's output socket, if there is one.
    pub coordinator_address: Option<String>,
    /// Set as `engine_index` on the outputs.
    pub engine_index: u32,
    /// Arrays of at least this many bytes go in auxiliary frames
    /// (`VLLM_MSGPACK_ZERO_COPY_THRESHOLD`).
    pub aux_frame_threshold: usize,
}

/// The socket of a message encoded by Python.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Destination {
    Client(usize),
    Coordinator,
}

enum Message {
    Batch {
        client_index: usize,
        outputs: RequestBatchOutputs,
    },
    Encoded {
        destination: Destination,
        frames: Vec<Bytes>,
    },
    /// Send `ENGINE_CORE_DEAD` to every client, then stop.
    Dead,
}

/// Called with the error that stopped the output thread.
pub type ErrorHandler = Box<dyn Fn(String) + Send>;

/// Handle to the output socket thread. Sending never blocks; the thread runs
/// until [`Self::send_dead`] or a socket error.
pub struct OutputWriter {
    tx: mpsc::UnboundedSender<Message>,
    engine_index: u32,
    stopped: Arc<Stopped>,
}

/// Whether the thread stopped, set as it exits.
#[derive(Default)]
struct Stopped {
    stopped: Mutex<bool>,
    condvar: Condvar,
}

/// Sets [`Stopped`] when dropped, however the thread exits.
struct StoppedGuard(Arc<Stopped>);

impl Drop for StoppedGuard {
    fn drop(&mut self) {
        *self.0.stopped.lock().unwrap() = true;
        self.0.condvar.notify_all();
    }
}

impl OutputWriter {
    /// Connect the output sockets and start the thread. Returns once the
    /// sockets are connected.
    pub fn start(config: OutputConfig, on_error: ErrorHandler) -> Result<Self, String> {
        let (tx, rx) = mpsc::unbounded_channel();
        let (connected_tx, connected_rx) = std_mpsc::channel();
        let stopped = Arc::new(Stopped::default());
        let guard = StoppedGuard(stopped.clone());
        let engine_index = config.engine_index;
        thread::Builder::new()
            .name("engine-core-output".to_string())
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
                    let sockets = match Sockets::connect(&config).await {
                        Ok(sockets) => {
                            let _ = connected_tx.send(Ok(()));
                            sockets
                        }
                        Err(error) => {
                            let _ = connected_tx.send(Err(error));
                            return;
                        }
                    };
                    if let Err(error) = run(sockets, rx, config.aux_frame_threshold).await {
                        on_error(error);
                    }
                });
            })
            .map_err(|error| error.to_string())?;
        match connected_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                tx,
                engine_index,
                stopped,
            }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err("the output thread stopped while connecting".to_string()),
        }
    }

    /// Send a batch of request outputs to `client_index`.
    pub fn send_batch(&self, client_index: usize, mut outputs: RequestBatchOutputs) {
        outputs.engine_index = self.engine_index;
        // After the thread stopped, outputs are dropped as Python's queue
        // would keep them unsent.
        let _ = self.tx.send(Message::Batch {
            client_index,
            outputs,
        });
    }

    /// Send a message Python encoded.
    pub fn send_encoded(&self, destination: Destination, frames: Vec<Bytes>) {
        let _ = self.tx.send(Message::Encoded {
            destination,
            frames,
        });
    }

    /// Send `ENGINE_CORE_DEAD` to every client, then stop the thread.
    pub fn send_dead(&self) {
        let _ = self.tx.send(Message::Dead);
    }

    /// Wait for the thread to stop, for at most `timeout` if given; returns
    /// whether it stopped.
    pub fn join(&self, timeout: Option<Duration>) -> bool {
        let stopped = self.stopped.stopped.lock().unwrap();
        let condvar = &self.stopped.condvar;
        let stopped = match timeout {
            Some(timeout) => condvar.wait_timeout_while(stopped, timeout, |s| !*s).unwrap().0,
            None => condvar.wait_while(stopped, |s| !*s).unwrap(),
        };
        *stopped
    }

    /// Whether the thread is still running.
    pub fn is_alive(&self) -> bool {
        !*self.stopped.stopped.lock().unwrap()
    }
}

struct Sockets {
    clients: Vec<PushSocket>,
    coordinator: Option<PushSocket>,
}

impl Sockets {
    async fn connect(config: &OutputConfig) -> Result<Self, String> {
        let connect = |address: String| async move {
            let mut socket = PushSocket::new();
            socket
                .connect(&address)
                .await
                .map_err(|error| format!("failed to connect output socket {address}: {error}"))?;
            Ok::<_, String>(socket)
        };
        let mut clients = Vec::with_capacity(config.client_addresses.len());
        for address in &config.client_addresses {
            clients.push(connect(address.clone()).await?);
        }
        let coordinator = match &config.coordinator_address {
            Some(address) => Some(connect(address.clone()).await?),
            None => None,
        };
        Ok(Self {
            clients,
            coordinator,
        })
    }

    fn get(&mut self, destination: Destination) -> Result<&mut PushSocket, String> {
        match destination {
            Destination::Client(index) => {
                let num_clients = self.clients.len();
                (self.clients.get_mut(index))
                    .ok_or_else(|| format!("client index {index} out of {num_clients} clients"))
            }
            Destination::Coordinator => {
                (self.coordinator.as_mut()).ok_or_else(|| "no DP coordinator".to_string())
            }
        }
    }
}

async fn send(socket: &mut PushSocket, frames: Vec<Bytes>) -> Result<(), String> {
    let message = ZmqMessage::try_from(frames).map_err(|error| error.to_string())?;
    socket.send(message).await.map_err(|error| error.to_string())
}

/// How long the sockets may take to deliver what was sent before they close,
/// as Python's `zmq.LINGER` of the output sockets.
const LINGER: Duration = Duration::from_secs(4);

/// Wait for the sockets' I/O tasks to finish writing what was sent.
///
/// A `zeromq` socket only queues a message for a background task to write, and
/// dropping the socket lets that task drain its queue and end; nothing would
/// write it once the runtime stops.
async fn linger(sockets: Sockets) {
    drop(sockets);
    let metrics = tokio::runtime::Handle::current().metrics();
    let deadline = tokio::time::Instant::now() + LINGER;
    while metrics.num_alive_tasks() > 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

async fn run(
    mut sockets: Sockets,
    mut rx: mpsc::UnboundedReceiver<Message>,
    aux_frame_threshold: usize,
) -> Result<(), String> {
    while let Some(message) = rx.recv().await {
        match message {
            Message::Batch {
                client_index,
                outputs,
            } => {
                let frames = encode_engine_core_outputs(
                    EngineCoreOutputs::RequestBatch(outputs),
                    aux_frame_threshold,
                )
                .map_err(|error| error.to_string())?;
                send(sockets.get(Destination::Client(client_index))?, frames).await?;
            }
            Message::Encoded {
                destination,
                frames,
            } => send(sockets.get(destination)?, frames).await?,
            Message::Dead => {
                for socket in &mut sockets.clients {
                    send(socket, vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)]).await?;
                }
                linger(sockets).await;
                return Ok(());
            }
        }
    }
    // The engine dropped the writer without `send_dead`.
    linger(sockets).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use vllm_engine_core_client::protocol::output::{EngineCoreOutput, decode_engine_core_outputs};
    use vllm_engine_core_client::test_utils::IpcNamespace;
    use zeromq::prelude::SocketRecv;
    use zeromq::{PullSocket, Socket as _};

    use super::*;

    async fn recv(socket: &mut PullSocket) -> Vec<Bytes> {
        socket.recv().await.unwrap().into_vec()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sends_batches_and_python_frames_in_order_then_dead() {
        let ipc = IpcNamespace::new().unwrap();
        let (client, coordinator) = (ipc.endpoint("client"), ipc.endpoint("coordinator"));
        let mut client_pull = PullSocket::new();
        client_pull.bind(&client).await.unwrap();
        let mut coordinator_pull = PullSocket::new();
        coordinator_pull.bind(&coordinator).await.unwrap();

        let writer = tokio::task::spawn_blocking(move || {
            OutputWriter::start(
                OutputConfig {
                    client_addresses: vec![client],
                    coordinator_address: Some(coordinator),
                    engine_index: 3,
                    aux_frame_threshold: 256,
                },
                Box::new(|error| panic!("{error}")),
            )
            .unwrap()
        })
        .await
        .unwrap();

        writer.send_encoded(Destination::Client(0), vec![Bytes::from_static(b"python")]);
        writer.send_batch(
            0,
            RequestBatchOutputs {
                outputs: vec![EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![7],
                    ..Default::default()
                }],
                finished_requests: Some(BTreeSet::new()),
                ..Default::default()
            },
        );
        writer.send_encoded(Destination::Coordinator, vec![Bytes::from_static(b"stats")]);
        writer.send_dead();

        assert_eq!(
            recv(&mut client_pull).await,
            [Bytes::from_static(b"python")]
        );
        let batch = decode_engine_core_outputs(&recv(&mut client_pull).await).unwrap();
        let batch = batch.into_request_batch().unwrap();
        assert_eq!(batch.engine_index, 3);
        assert_eq!(batch.outputs[0].new_token_ids, [7]);
        assert_eq!(
            recv(&mut client_pull).await,
            [Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)]
        );
        assert_eq!(
            recv(&mut coordinator_pull).await,
            [Bytes::from_static(b"stats")]
        );
        assert!(writer.join(Some(Duration::from_secs(5))));
        assert!(!writer.is_alive());
    }
}
