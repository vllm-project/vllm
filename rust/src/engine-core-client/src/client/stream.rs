// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ops::Deref;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::stream::FusedStream;
use futures::{Stream, StreamExt as _};
use itertools::Either;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::client::AbortRequest;
use crate::client::state::OutputReceiver;
use crate::protocol::output::{EngineCoreFinishReason, EngineCoreOutput};
use crate::{AbortCause, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Finished,
    ClosedWithError,
    UnexpectedClose,
}

/// One request-scoped engine-core output plus the enclosing batch metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineCoreStreamOutput {
    pub engine_index: u32,
    pub timestamp: f64,
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

/// Stream of raw engine-core outputs for one request.
///
/// The stream yields only [`EngineCoreStreamOutput`] values whose embedded
/// output `request_id` matches the originating `add_request()` call. Normal
/// request completion is expected to include a final output object whose
/// `finish_reason` is non-`None`.
pub struct EngineCoreOutputStream {
    request_id: String,
    engine_index: u32,
    abort_tx: mpsc::UnboundedSender<AbortRequest>,
    state: State,
    rx: OutputReceiver,
}

impl EngineCoreOutputStream {
    pub(crate) fn new(
        request_id: String,
        engine_index: u32,
        abort_tx: mpsc::UnboundedSender<AbortRequest>,
        rx: OutputReceiver,
    ) -> Self {
        Self {
            request_id,
            engine_index,
            abort_tx,
            state: State::Running,
            rx,
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

impl Stream for EngineCoreOutputStream {
    type Item = Result<EngineCoreStreamDelivery>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_terminated() {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => {
                match &item {
                    Ok(delivery) => {
                        // If the output indicates the request is finished, mark the stream as
                        // terminated with cleanly-finished state and expect no more outputs to
                        // come. Only the last output of a delivery can be terminal.
                        let output = delivery.last();
                        if output.finished() {
                            if output.finish_reason == Some(EngineCoreFinishReason::Error) {
                                error!(
                                    self.request_id,
                                    "request failed with an internal error during generation"
                                );
                            }
                            debug!(self.request_id, "request completed via final output");
                            self.state = State::Finished;
                        }
                    }
                    Err(error) => {
                        // If we get an error from the output stream, mark the stream as terminated
                        // with an error.
                        warn!(self.request_id, error = %error.as_report(), "request encountered an error");
                        self.state = State::ClosedWithError;
                    }
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                // If we get a `None` without seeing a finished output, this is an unexpected
                // close from the engine side. Mark the stream as terminated
                // with an unexpected close state and send an error down the
                // stream to notify the caller.
                warn!(self.request_id, "request stream closed unexpectedly");
                self.state = State::UnexpectedClose;

                Poll::Ready(Some(Err(Error::RequestStreamClosed {
                    request_id: self.request_id.clone(),
                })))
            }
        }
    }
}

impl FusedStream for EngineCoreOutputStream {
    fn is_terminated(&self) -> bool {
        !matches!(self.state, State::Running)
    }
}

impl Drop for EngineCoreOutputStream {
    fn drop(&mut self) {
        if self.is_terminated() {
            // If it's terminated, it means that the request either finished cleanly, or
            // encountered an error or unexpected close from the engine. In any
            // case, the request stream is already considered inactive and
            // there's no need to abort it on the engine side.
            return;
        }

        let abort_req = AbortRequest {
            request_id: self.request_id.clone(),
            cause: AbortCause::current(),
        };

        if self.abort_tx.send(abort_req).is_err() {
            warn!(
                request_id = self.request_id,
                "auto-abort worker already shut down; skip auto-abort"
            );
        }
    }
}
