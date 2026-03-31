use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use enum_as_inner::EnumAsInner;
use futures::Stream;
use futures::stream::FusedStream;
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::EngineCoreOutputStream;
use vllm_engine_core_client::protocol::{
    EngineCoreFinishReason, EngineCoreOutput, Logprobs, StopReason,
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

/// The reason a request finished.
///
/// This is a higher-level abstraction over engine-core's finish and stop reasons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, EnumAsInner)]
pub enum FinishReason {
    /// Generation stopped for a stop string, stop token, or EOS.
    ///
    /// The inner stop reason is present for explicit stop strings or stop tokens, and absent for
    /// EOS-driven stops.
    Stop(Option<StopReason>),
    /// `max_tokens` or `max_model_len` was reached.
    Length,
    /// The request was aborted by the client.
    Abort,
    /// A retryable request-level internal error occurred.
    Error,
    /// A repetitive token pattern was detected.
    Repetition,
}

impl FinishReason {
    /// Construct a stop finish reason caused by EOS rather than an explicit stop string/token.
    pub fn stop_eos() -> Self {
        Self::Stop(None)
    }

    /// Returns a human-readable string for this finish reason, used for metrics and reporting.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop(_) => "stop",
            Self::Length => "length",
            Self::Abort => "abort",
            Self::Error => "error",
            Self::Repetition => "repetition",
        }
    }

    /// If this is a stop finish reason, returns the inner stop reason if it exists.
    pub fn as_stop_reason(&self) -> Option<&StopReason> {
        match self {
            Self::Stop(stop_reason) => stop_reason.as_ref(),
            _ => None,
        }
    }

    /// If this is a stop finish reason, returns the inner stop reason if it exists.
    pub fn into_stop_reason(self) -> Option<StopReason> {
        match self {
            Self::Stop(stop_reason) => stop_reason,
            _ => None,
        }
    }
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
    /// Newly produced token IDs for this step.
    pub token_ids: Vec<u32>,
    /// Sample logprobs for the generated positions in this step.
    pub logprobs: Option<Logprobs>,

    /// Raw engine-core output.
    raw: EngineCoreOutput,
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

    /// Returns the finish reason when this output indicates terminal completion.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.raw.finish_reason.map(|reason| match reason {
            EngineCoreFinishReason::Stop => FinishReason::Stop(self.raw.stop_reason.clone()),
            EngineCoreFinishReason::Length => FinishReason::Length,
            EngineCoreFinishReason::Abort => FinishReason::Abort,
            EngineCoreFinishReason::Error => FinishReason::Error,
            EngineCoreFinishReason::Repetition => FinishReason::Repetition,
        })
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
    pending_prompt_info: Option<GeneratePromptInfo>,
    raw_stream: EngineCoreOutputStream,
    request_metrics: RequestMetricsTracker,
}

impl GenerateOutputStream {
    /// Create a new generate output stream by adapting one raw engine-core output stream.
    pub(crate) fn new(
        prompt_token_ids: Arc<[u32]>,
        raw_stream: EngineCoreOutputStream,
        request_metrics: RequestMetricsTracker,
    ) -> Self {
        Self {
            pending_prompt_info: Some(GeneratePromptInfo {
                prompt_token_ids,
                prompt_logprobs: None,
            }),
            raw_stream,
            request_metrics,
        }
    }
}

impl Stream for GenerateOutputStream {
    type Item = Result<GenerateOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
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

        // Populate the one-time prompt info on the first output.
        if let Some(info) = &mut self.pending_prompt_info
            && info.prompt_logprobs.is_none()
        {
            info.prompt_logprobs = raw.new_prompt_logprobs_tensors.as_deref().cloned();
        }

        let finished = raw.finished();
        let step_logprobs = raw.new_logprobs.as_deref().cloned();
        let output = GenerateOutput {
            request_id: raw.request_id.clone(),
            prompt_info: self.pending_prompt_info.take(),
            token_ids: raw.new_token_ids.clone(),
            logprobs: step_logprobs,
            raw,
        };

        if let Some(finish_reason) = output.finish_reason() {
            assert!(finished, "only finished outputs can have finish reasons");
            self.request_metrics
                .record_finished(received_at, finish_reason);
        }

        Poll::Ready(Some(Ok(output)))
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
