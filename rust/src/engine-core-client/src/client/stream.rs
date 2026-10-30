// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::stream::{BoxStream, FusedStream};
use futures::{Stream, StreamExt as _};
use itertools::Either;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use tracing::{debug, error, warn};

use crate::protocol::output::{EngineCoreFinishReason, EngineCoreOutput};
use crate::{Error, Result};

/// Output tokens `[start, end)` were sampled by weights labelled `version`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightVersionSpan {
    pub version: Arc<str>,
    pub start: usize,
    pub end: usize,
}

impl WeightVersionSpan {
    /// Spans over the output so far; `[]` for no tokens, `None` if no label was reported.
    pub(crate) fn from_starts(
        starts: &[(Arc<str>, usize)],
        num_output_tokens: usize,
    ) -> Option<Vec<Self>> {
        if starts.is_empty() && num_output_tokens > 0 {
            return None;
        }
        let ends = starts.iter().skip(1).map(|(_, start)| *start).chain([num_output_tokens]);
        Some(
            starts
                .iter()
                .zip(ends)
                .map(|((version, start), end)| Self {
                    version: version.clone(),
                    start: *start,
                    end,
                })
                .collect(),
        )
    }
}

/// Seconds on a process-wide monotonic clock, for producers that stamp
/// outputs as they receive them. Only differences between the outputs of one
/// request are meaningful, matching the engine's own batch timestamps.
pub fn monotonic_timestamp_secs() -> f64 {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// One request-scoped engine-core output, normalized across engine backends.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineCoreStreamOutput {
    /// The producing backend's monotonic timestamp; only differences
    /// between the outputs of one request are meaningful.
    pub timestamp: f64,
    /// Label of the weights that sampled this output, when the backend
    /// reports one per engine batch. The engine-core output channel turns
    /// these labels into `weight_versions` spans on the terminal output.
    pub weight_version: Option<Arc<str>>,
    /// Weight-version spans over the whole output. Set only on the terminal
    /// output; `None` when no version was reported for a non-empty output.
    pub weight_versions: Option<Vec<WeightVersionSpan>>,
    pub output: EngineCoreOutput,
}

impl Deref for EngineCoreStreamOutput {
    type Target = EngineCoreOutput;

    fn deref(&self) -> &Self::Target {
        &self.output
    }
}

/// Consecutive outputs of one request, delivered to its stream together and in
/// order. Outputs are grouped by the request's `stream_interval` (see
/// [`EngineCoreClient::call`]) but not merged; consumers merge them after
/// observing each raw output.
///
/// [`EngineCoreClient::call`]: crate::EngineCoreClient::call
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "single outputs stay inline, avoiding an allocation per engine step"
)]
pub enum EngineCoreStreamDelivery {
    /// A single output, sent without being held back.
    One(EngineCoreStreamOutput),
    /// Outputs held back by the stream interval, in order. Never empty.
    Many(Vec<EngineCoreStreamOutput>),
}

impl EngineCoreStreamDelivery {
    /// Return the first output.
    pub fn first(&self) -> &EngineCoreStreamOutput {
        match self {
            Self::One(output) => output,
            Self::Many(outputs) => outputs.first().expect("deliveries are never empty"),
        }
    }

    /// Return the last output, the only one that can be terminal.
    pub fn last(&self) -> &EngineCoreStreamOutput {
        match self {
            Self::One(output) => output,
            Self::Many(outputs) => outputs.last().expect("deliveries are never empty"),
        }
    }

    /// Iterate over the outputs in order.
    pub fn iter(
        &self,
    ) -> Either<
        std::iter::Once<&EngineCoreStreamOutput>,
        std::slice::Iter<'_, EngineCoreStreamOutput>,
    > {
        match self {
            Self::One(output) => Either::Left(std::iter::once(output)),
            Self::Many(outputs) => Either::Right(outputs.iter()),
        }
    }
}

impl IntoIterator for EngineCoreStreamDelivery {
    type Item = EngineCoreStreamOutput;
    type IntoIter =
        Either<std::iter::Once<EngineCoreStreamOutput>, std::vec::IntoIter<EngineCoreStreamOutput>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::One(output) => Either::Left(std::iter::once(output)),
            Self::Many(outputs) => Either::Right(outputs.into_iter()),
        }
    }
}

/// Stream of raw engine-core outputs for one request, over any engine
/// backend's raw delivery stream.
///
/// The stream yields only [`EngineCoreStreamOutput`] values whose embedded
/// output `request_id` matches the originating `add_request()` call. Normal
/// request completion is expected to include a final output object whose
/// `finish_reason` is non-`None`.
///
/// This adapter owns the [`EngineOutputStream`] contract shared by every
/// backend: a terminal output or an error terminates the stream, an inner
/// stream that ends first yields [`Error::RequestStreamClosed`], and dropping
/// a live stream runs `abort_on_drop` (when given) before the inner stream
/// itself is dropped.
///
/// [`EngineOutputStream`]: crate::EngineOutputStream
pub struct EngineOutputStreamAdapter {
    request_id: String,
    engine_index: u32,
    /// Dropped as soon as the stream terminates, so the backend releases
    /// the request's resources without waiting for this stream to drop.
    /// `None` marks the terminated state.
    inner: Option<BoxStream<'static, Result<EngineCoreStreamDelivery>>>,
    abort_on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl EngineOutputStreamAdapter {
    pub fn new(
        request_id: String,
        engine_index: u32,
        inner: impl Stream<Item = Result<EngineCoreStreamDelivery>> + Send + 'static,
        abort_on_drop: Option<Box<dyn FnOnce() + Send>>,
    ) -> Self {
        Self {
            request_id,
            engine_index,
            inner: Some(Box::pin(inner)),
            abort_on_drop,
        }
    }

    /// Return the engine-core `request_id` bound to this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Return the index of the engine that owns this request.
    pub fn engine_index(&self) -> u32 {
        self.engine_index
    }

    /// Flatten the stream into one item per raw output, for consumers that do
    /// not care how outputs are grouped into deliveries.
    pub fn into_outputs(self) -> impl Stream<Item = Result<EngineCoreStreamOutput>> + Send + Unpin {
        self.flat_map(|item| {
            futures::stream::iter(match item {
                Ok(delivery) => Either::Left(delivery.into_iter().map(Ok)),
                Err(error) => Either::Right(std::iter::once(Err(error))),
            })
        })
    }
}

impl Stream for EngineOutputStreamAdapter {
    type Item = Result<EngineCoreStreamDelivery>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(inner) = &mut self.inner else {
            return Poll::Ready(None);
        };

        match inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => {
                let terminated = match &item {
                    Ok(delivery) => {
                        // If the output indicates the request is finished, mark the stream as
                        // terminated with cleanly-finished state and expect no more outputs to
                        // come. Only the last output of a delivery can be terminal.
                        let output = delivery.last();
                        if output.finished() {
                            // A retryable error (e.g. a paused engine's rejection) is expected.
                            if output.finish_reason == Some(EngineCoreFinishReason::Error)
                                && !output.error.as_ref().is_some_and(|error| error.retryable)
                            {
                                error!(
                                    self.request_id,
                                    "request failed with an internal error during generation"
                                );
                            }
                            debug!(self.request_id, "request completed via final output");
                        }
                        output.finished()
                    }
                    Err(error) => {
                        // If we get an error from the output stream, mark the stream as terminated
                        // with an error.
                        warn!(self.request_id, error = %error.as_report(), "request encountered an error");
                        true
                    }
                };
                if terminated {
                    self.inner = None;
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                // If we get a `None` without seeing a finished output, this is an unexpected
                // close from the engine side. Mark the stream as terminated
                // with an unexpected close state and send an error down the
                // stream to notify the caller.
                warn!(self.request_id, "request stream closed unexpectedly");
                self.inner = None;

                Poll::Ready(Some(Err(Error::RequestStreamClosed {
                    request_id: self.request_id.clone(),
                })))
            }
        }
    }
}

impl FusedStream for EngineOutputStreamAdapter {
    fn is_terminated(&self) -> bool {
        self.inner.is_none()
    }
}

impl Drop for EngineOutputStreamAdapter {
    fn drop(&mut self) {
        if self.is_terminated() {
            // If it's terminated, it means that the request either finished cleanly, or
            // encountered an error or unexpected close from the engine. In any
            // case, the request stream is already considered inactive and
            // there's no need to abort it on the engine side.
            return;
        }

        if let Some(abort_on_drop) = self.abort_on_drop.take() {
            abort_on_drop();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures::StreamExt as _;

    use super::*;
    use crate::protocol::output::EngineCoreFinishReason;

    /// Sets its flag when dropped, proving the inner stream was released.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn output(finish_reason: Option<EngineCoreFinishReason>) -> Result<EngineCoreStreamDelivery> {
        Ok(EngineCoreStreamDelivery::One(EngineCoreStreamOutput {
            timestamp: 0.0,
            weight_version: None,
            weight_versions: None,
            output: EngineCoreOutput {
                request_id: "req".to_string(),
                finish_reason,
                ..EngineCoreOutput::default()
            },
        }))
    }

    fn adapter(
        items: Vec<Result<EngineCoreStreamDelivery>>,
        inner_dropped: Arc<AtomicBool>,
        aborted: Arc<AtomicBool>,
    ) -> EngineOutputStreamAdapter {
        let flag = DropFlag(inner_dropped);
        let mut items = items.into_iter();
        let inner = futures::stream::poll_fn(move |_| {
            let _ = &flag;
            Poll::Ready(items.next())
        });
        EngineOutputStreamAdapter::new(
            "req".to_string(),
            0,
            inner,
            Some(Box::new(move || aborted.store(true, Ordering::SeqCst))),
        )
    }

    #[tokio::test]
    async fn terminal_output_drops_inner_stream_without_aborting() {
        let inner_dropped = Arc::new(AtomicBool::new(false));
        let aborted = Arc::new(AtomicBool::new(false));
        let mut stream = adapter(
            vec![output(Some(EngineCoreFinishReason::Length))],
            inner_dropped.clone(),
            aborted.clone(),
        );

        assert!(stream.next().await.unwrap().is_ok());
        // Termination releases the backend's stream immediately, before
        // this stream itself is dropped.
        assert!(inner_dropped.load(Ordering::SeqCst));
        assert!(stream.next().await.is_none());
        drop(stream);
        assert!(!aborted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn dropping_a_live_stream_runs_the_abort_hook() {
        let inner_dropped = Arc::new(AtomicBool::new(false));
        let aborted = Arc::new(AtomicBool::new(false));
        let mut stream = adapter(vec![output(None)], inner_dropped, aborted.clone());

        assert!(stream.next().await.unwrap().is_ok());
        drop(stream);
        assert!(aborted.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn inner_stream_ending_early_yields_request_stream_closed() {
        let inner_dropped = Arc::new(AtomicBool::new(false));
        let aborted = Arc::new(AtomicBool::new(false));
        let mut stream = adapter(vec![], inner_dropped.clone(), aborted.clone());

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, Error::RequestStreamClosed { request_id } if request_id == "req"));
        assert!(inner_dropped.load(Ordering::SeqCst));
        assert!(stream.next().await.is_none());
        drop(stream);
        assert!(!aborted.load(Ordering::SeqCst));
    }
}
