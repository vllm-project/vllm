use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use futures::stream::FusedStream;
use thiserror_ext::AsReport as _;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::client::state::OutputReceiver;
use crate::protocol::EngineCoreOutput;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    Finished,
    ClosedWithError,
    UnexpectedClose,
}

/// Stream of raw engine-core outputs for one request.
///
/// The stream yields only [`EngineCoreOutput`] values whose `request_id` matches the originating
/// `add_request()` call. Normal request completion is expected to include a final output object
/// whose `finish_reason` is non-`None`.
pub struct EngineCoreOutputStream {
    request_id: String,
    abort_tx: mpsc::UnboundedSender<String>,
    state: State,
    rx: OutputReceiver,
}

impl EngineCoreOutputStream {
    pub(crate) fn new(
        request_id: String,
        abort_tx: mpsc::UnboundedSender<String>,
        rx: OutputReceiver,
    ) -> Self {
        Self {
            request_id,
            abort_tx,
            state: State::Running,
            rx,
        }
    }

    /// Return the engine-core `request_id` bound to this stream.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl Stream for EngineCoreOutputStream {
    type Item = Result<EngineCoreOutput>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_terminated() {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(item)) => {
                match &item {
                    Ok(output) => {
                        // If the output indicates the request is finished, mark the stream as
                        // terminated with cleanly-finished state and expect no more outputs to
                        // come.
                        if output.finished() {
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
                // If we get a `None` without seeing a finished output, this is an unexpected close
                // from the engine side. Mark the stream as terminated with an unexpected close
                // state and send an error down the stream to notify the caller.
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
            // If it's terminated, it means that the request either finished cleanly, or encountered
            // an error or unexpected close from the engine. In any case, the request stream is
            // already considered inactive and there's no need to abort it on the engine side.
            return;
        }

        let request_id = self.request_id.clone();
        if self.abort_tx.send(request_id.clone()).is_err() {
            warn!(
                request_id,
                "auto-abort worker already shut down; skip auto-abort"
            );
        }
    }
}
