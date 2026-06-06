use anyhow::{Context as _, Result, anyhow, bail};
use futures::{Stream, StreamExt as _, stream};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::warn;
use vllm_engine_core_client::mock_engine::MockEngineDataSockets;
use vllm_engine_core_client::protocol::utility::EngineCoreUtilityRequest;
use vllm_engine_core_client::protocol::{
    EngineCoreRequest, EngineCoreRequestType, decode_msgpack, encode_msgpack,
};
use zeromq::{DealerSocket, PushSocket, SocketRecv as _, SocketSend as _, ZmqMessage};

use crate::engine::{EngineInput, EngineOutput};

/// Send one engine output batch to the client over the appropriate push socket.
async fn send_engine_outputs_to_client(
    push_sockets: &mut [PushSocket],
    EngineOutput {
        client_index,
        outputs,
    }: EngineOutput,
) -> Result<()> {
    let message = ZmqMessage::from(encode_msgpack(&outputs)?);
    push_sockets[client_index as usize].send(message).await?;
    Ok(())
}

/// Create a stream of `EngineInput` by continuously receiving messages from the given dealer socket
/// and decoding them into `EngineInput`.
fn dealer_input_stream(dealer: DealerSocket) -> impl Stream<Item = Result<EngineInput>> {
    stream::unfold(dealer, |mut dealer| async {
        let input = loop {
            let message =
                match dealer.recv().await.context("failed to receive message from dealer socket") {
                    Ok(message) => message,
                    Err(err) => break Err(err),
                };

            match decode_request(message) {
                Ok(input) => break Ok(input),
                Err(err) => {
                    warn!(%err, "failed to decode engine request message; ignoring");
                }
            }
        };

        Some((input, dealer))
    })
}

/// Decode a `ZmqMessage` into an `EngineInput`. Returns an error if the message is malformed or
/// contains an unknown/unsupported request type.
fn decode_request(message: ZmqMessage) -> Result<EngineInput> {
    let frames = message.into_vec();
    if frames.is_empty() {
        bail!("empty engine request message");
    }
    if frames.len() != 2 {
        bail!("invalid frame count for engine request: {}", frames.len());
    }

    let request_type_frame = frames[0].as_ref();
    let Some(request_type) = EngineCoreRequestType::from_frame(request_type_frame) else {
        bail!("unknown engine request type: {:?}", request_type_frame);
    };

    let input = match request_type {
        EngineCoreRequestType::Add => {
            let request: Box<EngineCoreRequest> = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Request(request)
        }
        EngineCoreRequestType::Abort => {
            let request_ids: Vec<String> = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Abort(request_ids)
        }
        EngineCoreRequestType::Utility => {
            let request: EngineCoreUtilityRequest = decode_msgpack(frames[1].as_ref())?;
            EngineInput::Utility(request)
        }
        EngineCoreRequestType::StartDpWave => EngineInput::StartDpWave,
    };

    Ok(input)
}

/// Run the main IO loop for the mock engine, continuously receiving and decoding raw messages from
/// the dealer sockets, sending them to the engine loop task via `input_tx`, and receiving
/// `EngineOutput` from the engine loop task via `output_rx` and sending them to the client over the
/// appropriate push socket, until `shutdown` is cancelled.
pub(crate) async fn run_io_loop(
    data_sockets: Vec<MockEngineDataSockets>,
    input_tx: mpsc::UnboundedSender<EngineInput>,
    mut output_rx: mpsc::Receiver<EngineOutput>,
    shutdown: CancellationToken,
) -> Result<()> {
    let (dealers, mut push_sockets): (Vec<_>, Vec<_>) =
        data_sockets.into_iter().map(|sockets| (sockets.dealer, sockets.push)).unzip();
    let mut input_streams =
        stream::select_all(dealers.into_iter().map(dealer_input_stream).map(Box::pin));

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Ok(()),

            output = output_rx.recv() => {
                let output = output
                    .ok_or_else(|| anyhow!("mock engine output channel closed"))?;
                send_engine_outputs_to_client(&mut push_sockets, output).await?;
            }

            input = input_streams.next() => {
                let input = input
                    .ok_or_else(|| anyhow!("mock engine input streams closed"))??;
                input_tx
                    .send(input)
                    .map_err(|_| anyhow!("mock engine state task shut down"))?;
            }
        }
    }
}
