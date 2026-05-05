use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use enum_as_inner::EnumAsInner;
use futures::stream::FusedStream;
use futures::{Stream, StreamExt as _, pin_mut};
use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::{EngineCoreFinishReason, Logprobs, StopReason};
use vllm_engine_core_client::{AbortCause, EngineCoreOutputStream};

use crate::error::Result;
use crate::request_metrics::{RequestMetricsTracker, current_unix_timestamp_secs};

/// Final raw token output plus terminal stream metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectedGenerateOutput {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_logprobs: Option<Logprobs>,
    pub token_ids: Vec<u32>,
    pub logprobs: Option<Logprobs>,
    pub finish_reason: FinishReason,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
}

/// Prompt-scoped metadata emitted only once on the first [`GenerateOutput`] for
/// one request.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratePromptInfo {
    /// Original prompt token IDs for this request.
    pub prompt_token_ids: Arc<[u32]>,
    /// Prompt logprobs returned by engine-core for scored prompt positions,
    /// when requested.
    pub prompt_logprobs: Option<Logprobs>,
}

/// The reason a request finished.
///
/// This is a higher-level abstraction over engine-core's finish and stop
/// reasons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, EnumAsInner)]
pub enum FinishReason {
    /// Generation stopped for a stop string, stop token, or EOS.
    ///
    /// The inner stop reason is present for explicit stop strings or stop
    /// tokens, and absent for EOS-driven stops.
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
    /// Construct a stop finish reason caused by EOS rather than an explicit
    /// stop string/token.
    pub fn stop_eos() -> Self {
        Self::Stop(None)
    }

    /// Returns a human-readable string for this finish reason, used for metrics
    /// and reporting.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop(_) => "stop",
            Self::Length => "length",
            Self::Abort => "abort",
            Self::Error => "error",
            Self::Repetition => "repetition",
        }
    }

    /// If this is a stop finish reason, returns the inner stop reason if it
    /// exists.
    pub fn as_stop_reason(&self) -> Option<&StopReason> {
        match self {
            Self::Stop(stop_reason) => stop_reason.as_ref(),
            _ => None,
        }
    }

    /// If this is a stop finish reason, returns the inner stop reason if it
    /// exists.
    pub fn into_stop_reason(self) -> Option<StopReason> {
        match self {
            Self::Stop(stop_reason) => stop_reason,
            _ => None,
        }
    }
}

fn finish_reason_from_engine(
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
) -> Option<FinishReason> {
    finish_reason.map(|reason| match reason {
        EngineCoreFinishReason::Stop => FinishReason::Stop(stop_reason),
        EngineCoreFinishReason::Length => FinishReason::Length,
        EngineCoreFinishReason::Abort => FinishReason::Abort,
        EngineCoreFinishReason::Error => FinishReason::Error,
        EngineCoreFinishReason::Repetition => FinishReason::Repetition,
    })
}

/// Token and logprob output item returned by [`GenerateOutputStream`].
///
/// Original Python output reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/outputs.py#L85-L143>
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOutput {
    /// Unique ID of the request that produced this output.
    pub request_id: String,
    /// One-time prompt metadata emitted only on the first output for this
    /// request.
    pub prompt_info: Option<GeneratePromptInfo>,
    /// Newly produced token IDs for this step.
    pub token_ids: Vec<u32>,
    /// Sample logprobs for the generated positions in this step.
    pub logprobs: Option<Logprobs>,
    /// Terminal finish reason, when this is the final output for the request.
    pub finish_reason: Option<FinishReason>,
    /// Connector-specific KV transfer parameters for disaggregated serving.
    pub kv_transfer_params: Option<serde_json::Value>,
}

impl GenerateOutput {
    /// Returns the prompt token IDs when this output carries
    /// [`GeneratePromptInfo`].
    ///
    /// Only the first output for a request can return `Some`; all later outputs
    /// return `None`.
    pub fn prompt_token_ids(&self) -> Option<&Arc<[u32]>> {
        self.prompt_info.as_ref().map(|info| &info.prompt_token_ids)
    }

    /// Returns the prompt logprobs when this output carries
    /// [`GeneratePromptInfo`].
    ///
    /// Only the first output for a request can return `Some`; all later outputs
    /// return `None`.
    pub fn prompt_logprobs(&self) -> Option<&Logprobs> {
        self.prompt_info.as_ref().and_then(|info| info.prompt_logprobs.as_ref())
    }

    /// Returns whether this output is terminal for the request.
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

#[cfg(any(test, feature = "test-util"))]
impl GenerateOutput {
    /// Build a [`GenerateOutput`] for tests.
    pub fn for_test(
        prompt_token_ids: Option<Arc<[u32]>>,
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
    ) -> Self {
        Self {
            request_id: String::new(),
            prompt_info: prompt_token_ids.map(|ids| GeneratePromptInfo {
                prompt_token_ids: ids,
                prompt_logprobs: None,
            }),
            token_ids,
            logprobs: None,
            finish_reason,
            kv_transfer_params: None,
        }
    }
}

/// Stream of per-request generate outputs for one request.
///
/// - A normal termination of the stream represents a clean completion of the
///   request.
/// - For errors, unexpected closes, or explicit aborts, the stream terminates
///   with an error.
pub struct GenerateOutputStream {
    pending_prompt_info: Option<GeneratePromptInfo>,
    raw_stream: EngineCoreOutputStream,
    request_metrics: RequestMetricsTracker,
}

impl GenerateOutputStream {
    /// Create a new generate output stream by adapting one raw engine-core
    /// output stream.
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

    /// Return the internal engine request ID bound to this stream.
    pub fn request_id(&self) -> &str {
        self.raw_stream.request_id()
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
            info.prompt_logprobs =
                raw.new_prompt_logprobs_tensors.map(|value| value.into_direct().unwrap());
        }

        let logprobs = raw.new_logprobs.map(|value| value.into_direct().unwrap());

        let finish_reason = finish_reason_from_engine(raw.finish_reason, raw.stop_reason);
        if let Some(finish_reason) = finish_reason.as_ref() {
            self.request_metrics.record_finished(received_at, finish_reason.clone());
        }

        let output = GenerateOutput {
            request_id: raw.request_id,
            prompt_info: self.pending_prompt_info.take(),
            token_ids: raw.new_token_ids,
            logprobs,
            finish_reason,
            kv_transfer_params: raw.kv_transfer_params,
        };

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

        // If the user or the upper layer drops a live generate stream,
        // `EngineCoreOutputStream::Drop` will trigger an engine-side abort. Record the
        // matching terminal request metrics here so frontend-driven aborts are still
        // visible as `finished_reason=...` instead of disappearing from observability
        // entirely.
        let finish_reason = match AbortCause::current() {
            AbortCause::DroppedStream => FinishReason::Abort,
            AbortCause::StopStringMatched => FinishReason::Stop(None),
        };

        self.request_metrics
            .record_finished(current_unix_timestamp_secs(), finish_reason);
    }
}

#[allow(clippy::manual_async_fn, reason = "specify `Send` bound")]
#[easy_ext::ext(GenerateOutputStreamExt)]
impl<T: Stream<Item = Result<GenerateOutput>> + Send> T {
    /// Collect the raw generate stream to completion and return the final token
    /// output.
    pub fn collect_output(self) -> impl Future<Output = Result<CollectedGenerateOutput>> + Send {
        async move {
            let stream = self;
            pin_mut!(stream);
            let mut prompt_token_ids = None;
            let mut prompt_logprobs = None;
            let mut collected: Option<CollectedGenerateOutput> = None;

            while let Some(output) = stream.next().await.transpose()? {
                if let Some(info) = output.prompt_info {
                    if prompt_token_ids.is_none() {
                        prompt_token_ids = Some(info.prompt_token_ids.to_vec());
                    }
                    if prompt_logprobs.is_none() {
                        prompt_logprobs = info.prompt_logprobs;
                    }
                }

                if let Some(existing) = collected.as_mut() {
                    existing.token_ids.extend(output.token_ids);
                    if let Some(step_logprobs) = output.logprobs {
                        if let Some(collected_logprobs) = existing.logprobs.as_mut() {
                            collected_logprobs.positions.extend(step_logprobs.positions);
                        } else {
                            existing.logprobs = Some(step_logprobs);
                        }
                    }
                } else {
                    collected = Some(CollectedGenerateOutput {
                        request_id: output.request_id,
                        prompt_token_ids: prompt_token_ids.take().unwrap_or_default(),
                        prompt_logprobs: prompt_logprobs.take(),
                        token_ids: output.token_ids,
                        logprobs: output.logprobs,
                        finish_reason: FinishReason::Error,
                        kv_transfer_params: None,
                    });
                }

                if let Some(finish_reason) = output.finish_reason {
                    let mut collected = collected.expect("terminal output must exist");
                    collected.finish_reason = finish_reason;
                    collected.kv_transfer_params = output.kv_transfer_params;
                    return Ok(collected);
                }
            }

            unreachable!("generate stream should yield an error instead of closing early")
        }
    }
}
