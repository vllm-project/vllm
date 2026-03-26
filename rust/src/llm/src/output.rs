use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use futures::Stream;
use futures::stream::FusedStream;
use vllm_engine_core_client::EngineCoreOutputStream;
use vllm_engine_core_client::protocol::{
    EngineCoreOutput, FinishReason, Logprobs, PositionLogprobs, RequestOutputKind,
};

use crate::error::Result;
use crate::request_metrics::{RequestMetricsTracker, current_unix_timestamp_secs};

/// Prompt-scoped metadata emitted only once on the first [`GenerateOutput`] for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratePromptInfo {
    /// Original prompt token IDs for this request.
    pub prompt_token_ids: Arc<[u32]>,
    /// Prompt logprobs returned by engine-core for scored prompt positions, when requested.
    pub prompt_logprobs: Option<Logprobs>,
}

/// Token and logprob output item returned by [`GenerateOutputStream`].
///
/// Original Python output reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/outputs.py#L85-L143>
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    /// Unique ID of the request that produced this output.
    pub request_id: String,
    /// One-time prompt metadata emitted only on the first output for this request.
    pub prompt_info: Option<GeneratePromptInfo>,
    /// Generated token IDs for this update.
    ///
    /// The exact semantics depend on the request's `output_kind`:
    /// - `Delta`: only the newly produced token IDs for this step
    /// - `FinalOnly`: the full completion, emitted once on the terminal step
    pub token_ids: Vec<u32>,
    /// Sample logprobs for the generated positions in this update.
    ///
    /// For `Delta`, this is the per-step payload returned by engine-core. For `FinalOnly`, this
    /// accumulates all generated positions and is emitted once on the terminal step.
    pub logprobs: Option<Logprobs>,
    /// Raw engine-core output for callers that need finish reason, stop reason, or other
    /// engine-native fields.
    pub raw: EngineCoreOutput,
}

impl GenerateOutput {
    /// Returns the prompt token IDs when this output carries [`GeneratePromptInfo`].
    ///
    /// Only the first output for a request can return `Some`; all later outputs return `None`.
    pub fn prompt_token_ids(&self) -> Option<&[u32]> {
        self.prompt_info
            .as_ref()
            .map(|info| info.prompt_token_ids.as_ref())
    }

    /// Returns the prompt logprobs when this output carries [`GeneratePromptInfo`].
    ///
    /// Only the first output for a request can return `Some`; all later outputs return `None`.
    pub fn prompt_logprobs(&self) -> Option<&Logprobs> {
        self.prompt_info
            .as_ref()
            .and_then(|info| info.prompt_logprobs.as_ref())
    }

    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.raw.finished()
    }
}

/// Stream of per-request generate outputs for one request.
///
/// - A normal termination of the stream represents a clean completion of the request.
/// - For errors, unexpected closes, or explicit aborts, the stream terminates with an error.
pub struct GenerateOutputStream {
    output_kind: RequestOutputKind,
    pending_prompt_info: Option<GeneratePromptInfo>,
    raw_stream: EngineCoreOutputStream,
    request_metrics: RequestMetricsTracker,
    collected_token_ids: Vec<u32>,
    collected_logprob_positions: Option<Vec<PositionLogprobs>>,
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
            pending_prompt_info: Some(GeneratePromptInfo {
                prompt_token_ids,
                prompt_logprobs: None,
            }),
            raw_stream,
            request_metrics,
            collected_token_ids: Vec::new(),
            collected_logprob_positions: None,
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
            let finish_reason = raw.finish_reason;

            // Populate the one-time prompt info on the first output.
            if let Some(info) = &mut self.pending_prompt_info
                && info.prompt_logprobs.is_none()
            {
                info.prompt_logprobs = raw.new_prompt_logprobs_tensors.as_deref().cloned();
            }

            let finished = raw.finished();
            let step_logprobs = raw.new_logprobs.as_deref().cloned();
            let output = match self.output_kind {
                RequestOutputKind::Delta => Some(GenerateOutput {
                    request_id: raw.request_id.clone(),
                    prompt_info: self.pending_prompt_info.take(),
                    token_ids: raw.new_token_ids.clone(),
                    logprobs: step_logprobs,
                    raw,
                }),
                RequestOutputKind::FinalOnly => {
                    self.collected_token_ids
                        .extend_from_slice(&raw.new_token_ids);
                    if let Some(step_logprobs) = step_logprobs {
                        self.collected_logprob_positions
                            .get_or_insert_with(Vec::new)
                            .extend(step_logprobs.positions);
                    }
                    // `FINAL_ONLY` suppresses intermediate updates and emits once when the
                    // underlying raw output indicates terminal completion.
                    finished.then(|| GenerateOutput {
                        request_id: raw.request_id.clone(),
                        prompt_info: self.pending_prompt_info.take(),
                        token_ids: std::mem::take(&mut self.collected_token_ids),
                        logprobs: self
                            .collected_logprob_positions
                            .take()
                            .map(|positions| Logprobs { positions }),
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
