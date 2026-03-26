use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use futures::Stream;
use futures::stream::FusedStream;
use vllm_engine_core_client::EngineCoreOutputStream;
use vllm_engine_core_client::protocol::{EngineCoreOutput, FinishReason, RequestOutputKind};

use crate::error::Result;
use crate::request_metrics::{RequestMetricsTracker, current_unix_timestamp_secs};

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
    /// The exact semantics depend on the request's `output_kind`:
    /// - `Delta`: only the newly produced token IDs for this step
    /// - `FinalOnly`: the full completion, emitted once on the terminal step
    pub token_ids: Vec<u32>,
    /// Raw engine-core output for callers that need finish reason, stop reason, or other
    /// engine-native fields.
    pub raw: EngineCoreOutput,
}

impl GenerateOutput {
    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.raw.finished()
    }
}

/// Stream of token-only generate outputs for one request.
///
/// - A normal termination of the stream represents a clean completion of the request.
/// - For errors, unexpected closes, or explicit aborts, the stream terminates with an error.
pub struct GenerateOutputStream {
    output_kind: RequestOutputKind,
    prompt_token_ids: Arc<[u32]>,
    raw_stream: EngineCoreOutputStream,
    collected_token_ids: Vec<u32>,
    request_metrics: RequestMetricsTracker,
}

impl GenerateOutputStream {
    /// Create a new generate output stream by adapting one raw engine-core output stream.
    pub(crate) fn new(
        output_kind: RequestOutputKind,
        prompt_token_ids: Arc<[u32]>,
        raw_stream: EngineCoreOutputStream,
        request_metrics: RequestMetricsTracker,
    ) -> Self {
        Self {
            output_kind,
            prompt_token_ids,
            raw_stream,
            collected_token_ids: Vec::new(),
            request_metrics,
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

            let received_at = current_unix_timestamp_secs();
            self.request_metrics.observe_output(
                raw.engine_index,
                raw.timestamp,
                received_at,
                &raw.output,
            );

            let raw = raw.output;
            let finished = raw.finished();
            let finish_reason = raw.finish_reason;

            let output = match self.output_kind {
                RequestOutputKind::Delta => Some(GenerateOutput {
                    request_id: raw.request_id.clone(),
                    prompt_token_ids: self.prompt_token_ids.clone(),
                    token_ids: raw.new_token_ids.clone(),
                    raw,
                }),
                RequestOutputKind::FinalOnly => {
                    self.collected_token_ids
                        .extend_from_slice(&raw.new_token_ids);
                    // `FINAL_ONLY` suppresses intermediate updates and emits once when the
                    // underlying raw output indicates terminal completion.
                    finished.then(|| GenerateOutput {
                        request_id: raw.request_id.clone(),
                        prompt_token_ids: self.prompt_token_ids.clone(),
                        token_ids: std::mem::take(&mut self.collected_token_ids),
                        raw,
                    })
                }
            };

            if let Some(finish_reason) = finish_reason {
                assert!(finished, "only finished outputs can have finish reasons");
                self.request_metrics
                    .record_finished(received_at, finish_reason);
            }

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

impl Drop for GenerateOutputStream {
    fn drop(&mut self) {
        if self.raw_stream.is_terminated() {
            // Already terminated cleanly, no need to record abort metrics.
            return;
        }

        // If the user drops a live generate stream, `EngineCoreOutputStream::Drop` will trigger an
        // engine-side abort. Record the matching terminal request metrics here so frontend-driven
        // aborts are still visible as `finished_reason="abort"` instead of disappearing from
        // observability entirely.
        self.request_metrics
            .record_finished(current_unix_timestamp_secs(), FinishReason::Abort);
    }
}
