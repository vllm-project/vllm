// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-request output channel between the output dispatcher and
//! [`EngineCoreOutputStream`](crate::EngineCoreOutputStream).
//!
//! With a stream interval above one, the sender holds back the outputs after
//! the first one until they carry at least that many new tokens, then delivers
//! them in one channel message. The request task is then woken once per
//! delivery instead of once per engine step. Outputs are never merged,
//! reordered, or dropped; the receiver still yields them one by one.

use std::num::NonZeroU32;
use std::task::{Context, Poll, ready};

use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::client::stream::EngineCoreStreamOutput;
use crate::error::{Error, Result};

/// Create the output channel for one request.
pub(crate) fn output_channel(stream_interval: NonZeroU32) -> (OutputSender, OutputReceiver) {
    let (tx, rx) = mpsc::unbounded_channel();
    let sender = OutputSender {
        tx,
        stream_interval: stream_interval.get() as usize,
        delivered_first: false,
        buffered: Vec::new(),
        buffered_tokens: 0,
    };
    let receiver = OutputReceiver {
        rx,
        ready: Vec::new().into_iter(),
    };
    (sender, receiver)
}

/// One channel message, carrying outputs of the same request in order.
#[derive(Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "unbatched outputs stay inline, avoiding an allocation per engine step"
)]
enum Delivery {
    One(EngineCoreStreamOutput),
    Batch(Vec<EngineCoreStreamOutput>),
}

/// Sending half of a request's output channel, owned by its registry entry.
#[derive(Debug)]
pub(crate) struct OutputSender {
    tx: mpsc::UnboundedSender<Result<Delivery>>,
    /// Minimum number of new tokens per delivery after the first one.
    stream_interval: usize,
    /// Whether the first output has been delivered.
    delivered_first: bool,
    /// Outputs held back since the last delivery.
    buffered: Vec<EngineCoreStreamOutput>,
    /// Number of new tokens in `buffered`.
    buffered_tokens: usize,
}

impl OutputSender {
    /// Send one output, or hold it back until the held-back outputs reach the
    /// stream interval. The first output and the terminal output are always
    /// delivered immediately, the latter together with any held-back outputs.
    pub fn send(&mut self, output: EngineCoreStreamOutput) {
        let finished = output.finished();
        if self.buffered.is_empty()
            && (finished || !self.delivered_first || self.stream_interval == 1)
        {
            self.delivered_first = true;
            self.deliver(Delivery::One(output));
            return;
        }

        self.buffered_tokens += output.new_token_ids.len();
        self.buffered.push(output);
        if finished || self.buffered_tokens >= self.stream_interval {
            self.flush();
        }
    }

    /// Deliver the held-back outputs, if any.
    pub fn flush(&mut self) {
        if self.buffered.is_empty() {
            return;
        }
        self.buffered_tokens = 0;
        let outputs = std::mem::take(&mut self.buffered);
        self.deliver(Delivery::Batch(outputs));
    }

    /// Deliver the held-back outputs, then terminate the stream with `error`.
    pub fn send_error(mut self, error: Error) {
        self.flush();
        let _ = self.tx.send(Err(error));
    }

    fn deliver(&self, delivery: Delivery) {
        if let Err(mpsc::error::SendError(Ok(delivery))) = self.tx.send(Ok(delivery)) {
            let request_id = match &delivery {
                Delivery::One(output) => &output.request_id,
                Delivery::Batch(outputs) => &outputs[0].request_id,
            };
            debug!(request_id, "request output stream receiver dropped");
        }
    }
}

impl Drop for OutputSender {
    fn drop(&mut self) {
        // Every path that ends a live request delivers held-back outputs first
        // (terminal output, `flush`, or `send_error`). Outputs left here are lost.
        if !self.buffered.is_empty() && !self.tx.is_closed() {
            warn!(
                request_id = self.buffered[0].request_id,
                count = self.buffered.len(),
                "dropping request output sender with held-back outputs"
            );
        }
    }
}

/// Receiving half of a request's output channel. Yields outputs one by one.
#[derive(Debug)]
pub(crate) struct OutputReceiver {
    rx: mpsc::UnboundedReceiver<Result<Delivery>>,
    /// Outputs of the last batched delivery that have not been yielded yet.
    ready: std::vec::IntoIter<EngineCoreStreamOutput>,
}

impl OutputReceiver {
    /// Poll for the next output. Outputs of a batched delivery are yielded
    /// without returning `Pending` in between.
    pub fn poll_recv(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<EngineCoreStreamOutput>>> {
        if let Some(output) = self.ready.next() {
            return Poll::Ready(Some(Ok(output)));
        }
        let item = match ready!(self.rx.poll_recv(cx)) {
            Some(Ok(Delivery::One(output))) => Some(Ok(output)),
            Some(Ok(Delivery::Batch(outputs))) => {
                self.ready = outputs.into_iter();
                self.ready.next().map(Ok)
            }
            Some(Err(error)) => Some(Err(error)),
            None => None,
        };
        Poll::Ready(item)
    }
}

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::num::NonZeroU32;

    use expect_test::expect;
    use tokio::sync::mpsc::error::TryRecvError;

    use super::*;
    use crate::protocol::output::{EngineCoreFinishReason, EngineCoreOutput};

    fn output(token_ids: &[u32]) -> EngineCoreStreamOutput {
        EngineCoreStreamOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: EngineCoreOutput {
                request_id: "req".to_string(),
                new_token_ids: token_ids.to_vec(),
                ..EngineCoreOutput::default()
            },
        }
    }

    fn finished(token_ids: &[u32]) -> EngineCoreStreamOutput {
        let mut output = output(token_ids);
        output.output.finish_reason = Some(EngineCoreFinishReason::Length);
        output
    }

    fn channel(stream_interval: u32) -> (OutputSender, OutputReceiver) {
        output_channel(NonZeroU32::new(stream_interval).unwrap())
    }

    /// Drain all pending channel messages, rendering each delivery as the
    /// token ids of its outputs.
    fn deliveries(receiver: &mut OutputReceiver) -> Vec<Vec<Vec<u32>>> {
        let mut deliveries = Vec::new();
        loop {
            match receiver.rx.try_recv() {
                Ok(Ok(Delivery::One(output))) => deliveries.push(vec![output.output.new_token_ids]),
                Ok(Ok(Delivery::Batch(outputs))) => deliveries
                    .push(outputs.into_iter().map(|output| output.output.new_token_ids).collect()),
                Ok(Err(error)) => panic!("unexpected error delivery: {error:?}"),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return deliveries,
            }
        }
    }

    #[test]
    fn stream_interval_one_delivers_every_output() {
        let (mut sender, mut receiver) = channel(1);
        sender.send(output(&[1]));
        sender.send(output(&[2, 3]));
        sender.send(finished(&[4]));

        expect![[r#"
            [
                [
                    [
                        1,
                    ],
                ],
                [
                    [
                        2,
                        3,
                    ],
                ],
                [
                    [
                        4,
                    ],
                ],
            ]
        "#]]
        .assert_debug_eq(&deliveries(&mut receiver));
    }

    #[test]
    fn stream_interval_batches_outputs_after_the_first() {
        let (mut sender, mut receiver) = channel(3);
        // The first output is delivered immediately; later ones are held back
        // until they carry at least 3 tokens, counted across multi-token
        // outputs. The terminal output flushes whatever is held back.
        for token_ids in [&[1][..], &[2], &[3], &[4, 5], &[6], &[7, 8, 9, 10], &[11]] {
            sender.send(output(token_ids));
        }
        sender.send(finished(&[12]));

        expect![[r#"
            [
                [
                    [
                        1,
                    ],
                ],
                [
                    [
                        2,
                    ],
                    [
                        3,
                    ],
                    [
                        4,
                        5,
                    ],
                ],
                [
                    [
                        6,
                    ],
                    [
                        7,
                        8,
                        9,
                        10,
                    ],
                ],
                [
                    [
                        11,
                    ],
                    [
                        12,
                    ],
                ],
            ]
        "#]]
        .assert_debug_eq(&deliveries(&mut receiver));
    }

    #[test]
    fn terminal_output_without_held_back_outputs_is_delivered_alone() {
        let (mut sender, mut receiver) = channel(2);
        sender.send(output(&[1]));
        sender.send(output(&[2, 3]));
        sender.send(finished(&[]));

        expect![[r#"
            [
                [
                    [
                        1,
                    ],
                ],
                [
                    [
                        2,
                        3,
                    ],
                ],
                [
                    [],
                ],
            ]
        "#]]
        .assert_debug_eq(&deliveries(&mut receiver));
    }

    #[test]
    fn flush_delivers_held_back_outputs() {
        let (mut sender, mut receiver) = channel(4);
        sender.send(output(&[1]));
        sender.send(output(&[2]));
        sender.flush();
        sender.flush();

        expect![[r#"
            [
                [
                    [
                        1,
                    ],
                ],
                [
                    [
                        2,
                    ],
                ],
            ]
        "#]]
        .assert_debug_eq(&deliveries(&mut receiver));
    }

    #[test]
    fn error_follows_held_back_outputs() {
        let (mut sender, mut receiver) = channel(4);
        sender.send(output(&[1]));
        sender.send(output(&[2]));
        sender.send_error(Error::EngineCoreDead);

        let mut items = Vec::new();
        while let Ok(item) = receiver.rx.try_recv() {
            items.push(match item {
                Ok(Delivery::One(_)) => "one",
                Ok(Delivery::Batch(_)) => "batch",
                Err(_) => "error",
            });
        }
        assert_eq!(items, ["one", "batch", "error"]);
    }

    #[tokio::test]
    async fn receiver_yields_batched_outputs_one_by_one() {
        let (mut sender, mut receiver) = channel(2);
        sender.send(output(&[1]));
        sender.send(output(&[2]));
        sender.send(finished(&[3]));
        drop(sender);

        let mut token_ids = Vec::new();
        while let Some(item) = poll_fn(|cx| receiver.poll_recv(cx)).await {
            token_ids.push(item.unwrap().output.new_token_ids);
        }
        expect![[r#"
            [
                [
                    1,
                ],
                [
                    2,
                ],
                [
                    3,
                ],
            ]
        "#]]
        .assert_debug_eq(&token_ids);
    }
}
