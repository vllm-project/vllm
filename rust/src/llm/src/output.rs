use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use futures::Stream;
use futures::stream::FusedStream;
use vllm_engine_core_client::EngineCoreOutputStream;
use vllm_engine_core_client::protocol::{EngineCoreOutput, RequestOutputKind};

use crate::error::Result;

/// Token-only output item returned by [`GenerateOutputStream`].
///
/// Original Python output reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/outputs.py#L85-L143>
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    /// Unique ID of the request that produced this output.
    pub request_id: String,
    /// Original prompt token IDs for the request.
    pub prompt_token_ids: Arc<[u32]>,
    /// Generated token IDs for this update.
    ///
    /// The exact semantics depend on `sampling_params.output_kind`:
    /// - `Delta`: only the newly produced token IDs for this step
    /// - `Cumulative`: the full completion-so-far
    /// - `FinalOnly`: the full completion, emitted once on the terminal step
    pub token_ids: Vec<u32>,
    /// Raw engine-core output for callers that need finish reason, stop reason, or other
    /// engine-native fields.
    pub raw: EngineCoreOutput,
}

/// Stream of token-only generate outputs for one request.
///
/// - A normal termination of the stream represents a clean completion of the request.
/// - For errors, unexpected closes, or explicit aborts, the stream terminates with an error.
pub struct GenerateOutputStream {
    output_kind: RequestOutputKind,
    prompt_token_ids: Arc<[u32]>,
    raw_stream: EngineCoreOutputStream,
    cumulative_token_ids: Vec<u32>,
}

impl GenerateOutputStream {
    /// Create a new generate output stream by adapting one raw engine-core output stream.
    pub(crate) fn new(
        output_kind: RequestOutputKind,
        prompt_token_ids: Arc<[u32]>,
        raw_stream: EngineCoreOutputStream,
    ) -> Self {
        Self {
            output_kind,
            prompt_token_ids,
            raw_stream,
            cumulative_token_ids: Vec::new(),
        }
    }
}

impl Stream for GenerateOutputStream {
    type Item = Result<GenerateOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let raw = match ready!(Pin::new(&mut self.raw_stream).poll_next(cx)) {
                Some(Ok(raw)) => raw,
                Some(Err(error)) => return Poll::Ready(Some(Err(error.into()))),
                None => return Poll::Ready(None),
            };
            let finished = raw.finished();

            let output = match self.output_kind {
                RequestOutputKind::Delta => Some(GenerateOutput {
                    request_id: raw.request_id.clone(),
                    prompt_token_ids: self.prompt_token_ids.clone(),
                    token_ids: raw.new_token_ids.clone(),
                    raw,
                }),
                RequestOutputKind::Cumulative => {
                    self.cumulative_token_ids
                        .extend_from_slice(&raw.new_token_ids);
                    Some(GenerateOutput {
                        request_id: raw.request_id.clone(),
                        prompt_token_ids: self.prompt_token_ids.clone(),
                        token_ids: self.cumulative_token_ids.clone(),
                        raw,
                    })
                }
                RequestOutputKind::FinalOnly => {
                    self.cumulative_token_ids
                        .extend_from_slice(&raw.new_token_ids);
                    // `FINAL_ONLY` suppresses intermediate updates and emits once when the
                    // underlying raw output indicates terminal completion.
                    finished.then(|| GenerateOutput {
                        request_id: raw.request_id.clone(),
                        prompt_token_ids: self.prompt_token_ids.clone(),
                        token_ids: self.cumulative_token_ids.clone(),
                        raw,
                    })
                }
            };

            if let Some(output) = output {
                return Poll::Ready(Some(Ok(output)));
            }
        }
    }
}

impl FusedStream for GenerateOutputStream {
    fn is_terminated(&self) -> bool {
        self.raw_stream.is_terminated()
    }
}
