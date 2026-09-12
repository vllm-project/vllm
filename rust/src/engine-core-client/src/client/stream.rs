// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ops::Deref;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use futures::stream::FusedStream;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::client::AbortRequest;
use crate::client::state::{OutputReceiver, StreamEvent};
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

/// Stream of raw engine-core outputs for one request.
///
/// One-shot streams end on a terminal `finish_reason`. Resumable streams also
/// end on an explicit [`StreamEvent::SessionFinished`] from the registry.
pub struct EngineCoreOutputStream {
    request_id: String,
    engine_index: u32,
    abort_tx: mpsc::UnboundedSender<AbortRequest>,
    state: State,
    rx: OutputReceiver,
    resumable: bool,
}

impl EngineCoreOutputStream {
    pub(crate) fn new(
        request_id: String,
        engine_index: u32,
        abort_tx: mpsc::UnboundedSender<AbortRequest>,
        rx: OutputReceiver,
        resumable: bool,
    ) -> Self {
        Self {
            request_id,
            engine_index,
            abort_tx,
            state: State::Running,
            rx,
            resumable,
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
}

impl Stream for EngineCoreOutputStream {
    type Item = Result<EngineCoreStreamOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_terminated() {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(StreamEvent::Output(output)))) => {
                let output = *output;
                if output.finish_reason == Some(EngineCoreFinishReason::Error) {
                    error!(
                        self.request_id,
                        "request failed with an internal error during generation"
                    );
                }
                if output.finished() && !self.resumable {
                    debug!(self.request_id, "request completed via final output");
                    self.state = State::Finished;
                }
                Poll::Ready(Some(Ok(output)))
            }
            Poll::Ready(Some(Ok(StreamEvent::SessionFinished))) => {
                debug!(self.request_id, "request completed via registry signal");
                self.state = State::Finished;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                warn!(self.request_id, error = %error.as_report(), "request encountered an error");
                self.state = State::ClosedWithError;
                Poll::Ready(Some(Err(error)))
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

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use crate::client::state::StreamEvent;

    use super::*;

    fn segment_stop() -> EngineCoreStreamOutput {
        EngineCoreStreamOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: EngineCoreOutput {
                request_id: "rt-abc".to_string(),
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            },
        }
    }

    /// A segment stop leaves a resumable stream running, so losing the sender
    /// afterwards must surface an error rather than a silently truncated
    /// transcript.
    #[tokio::test]
    async fn resumable_sender_drop_after_segment_finish_is_unexpected() {
        let (abort_tx, _abort_rx) = mpsc::unbounded_channel();
        let (tx, rx) = mpsc::unbounded_channel();
        let mut stream = EngineCoreOutputStream::new("rt-abc".to_string(), 0, abort_tx, rx, true);

        tx.send(Ok(StreamEvent::Output(Box::new(segment_stop())))).unwrap();
        let segment = stream.next().await.unwrap().unwrap();
        assert_eq!(segment.finish_reason, Some(EngineCoreFinishReason::Stop));

        drop(tx);
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            Error::RequestStreamClosed { request_id } if request_id == "rt-abc"
        ));
    }

    /// Explicit completion has to terminate the stream, not merely end the
    /// iteration: a stream left running auto-aborts a request the engine has
    /// already finished.
    #[tokio::test]
    async fn resumable_completion_terminates_the_stream_without_aborting() {
        let (abort_tx, mut abort_rx) = mpsc::unbounded_channel();
        let (tx, rx) = mpsc::unbounded_channel();
        let mut stream = EngineCoreOutputStream::new("rt-abc".to_string(), 0, abort_tx, rx, true);

        tx.send(Ok(StreamEvent::Output(Box::new(segment_stop())))).unwrap();
        tx.send(Ok(StreamEvent::SessionFinished)).unwrap();

        assert!(stream.next().await.unwrap().is_ok());
        assert!(stream.next().await.is_none());
        assert!(stream.is_terminated());

        drop(stream);
        assert!(
            abort_rx.try_recv().is_err(),
            "a completed session must not be aborted"
        );
    }
}
