// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-request output channel between the output dispatcher and
//! [`EngineCoreOutputStream`](crate::EngineCoreOutputStream).
//!
//! With a stream interval above one, the sender holds back the outputs after
//! the first one until they carry at least that many new tokens, then delivers
//! them in one channel message, which the stream yields as one
//! [`EngineCoreStreamDelivery`](crate::EngineCoreStreamDelivery). The request
//! task is then woken once per delivery instead of once per engine step.
//!
//! Outputs are never merged, reordered, or dropped here; the consumer merges a
//! delivery into one output (see `vllm-llm`). Merging here would hide the raw
//! outputs that request metrics observe one by one (per-step timestamps for
//! inter-token latency, prefill stats, lifecycle events), and would have to
//! combine wire-level fields such as undecoded logprobs tensors and opaque
//! values.

use std::num::NonZeroU32;

use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::client::stream::{EngineCoreStreamDelivery as Delivery, EngineCoreStreamOutput};
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
    (sender, rx)
}

/// Receiving half of a request's output channel.
pub(crate) type OutputReceiver = mpsc::UnboundedReceiver<Result<Delivery>>;

/// Sending half of a request's output channel, owned by its registry entry.
#[derive(Debug)]
pub(crate) struct OutputSender {
    tx: mpsc::UnboundedSender<Result<Delivery>>,
    /// Minimum number of new tokens per delivery after the first one.
    stream_interval: usize,
    /// Whether the first ordinary output has been delivered.
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
    /// Progress snapshots also bypass the token interval, since prefill may
    /// produce no tokens for many steps. A progress-only output does not use
    /// up the immediate delivery of the first generated token.
    pub fn send(&mut self, output: EngineCoreStreamOutput) {
        let finished = output.finished();
        let has_progress = output.prefill_progress.is_some();
        if self.buffered.is_empty()
            && (finished || has_progress || !self.delivered_first || self.stream_interval == 1)
        {
            self.delivered_first |= !has_progress
                || !output.new_token_ids.is_empty()
                || output.pooling_output.is_some()
                || finished;
            self.deliver(Delivery::One(output));
            return;
        }

        self.buffered_tokens += output.new_token_ids.len();
        self.buffered.push(output);
        if finished || has_progress || self.buffered_tokens >= self.stream_interval {
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
        self.deliver(Delivery::Many(outputs));
    }

    /// Deliver the held-back outputs, then terminate the stream with `error`.
    pub fn send_error(mut self, error: Error) {
        self.flush();
        let _ = self.tx.send(Err(error));
    }

    fn deliver(&self, delivery: Delivery) {
        if let Err(mpsc::error::SendError(Ok(delivery))) = self.tx.send(Ok(delivery)) {
            debug!(
                request_id = delivery.first().request_id,
                "request output stream receiver dropped"
            );
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use expect_test::expect;
    use tokio::sync::mpsc::error::TryRecvError;

    use super::*;
    use crate::protocol::output::{EngineCoreFinishReason, EngineCoreOutput, PrefillProgress};

    fn output(token_ids: &[u32]) -> EngineCoreStreamOutput {
        EngineCoreStreamOutput {
            engine_index: 0,
            timestamp: 0.0,
            weight_version: None,
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

    /// Render a delivery as the token ids of its outputs.
    fn token_ids(delivery: Delivery) -> Vec<Vec<u32>> {
        delivery.into_iter().map(|output| output.output.new_token_ids).collect()
    }

    /// Drain all pending channel messages, rendering each delivery as the
    /// token ids of its outputs.
    fn deliveries(receiver: &mut OutputReceiver) -> Vec<Vec<Vec<u32>>> {
        let mut deliveries = Vec::new();
        loop {
            match receiver.try_recv() {
                Ok(Ok(delivery)) => deliveries.push(token_ids(delivery)),
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
    fn prefill_progress_does_not_wait_for_generated_tokens() {
        let (mut sender, mut receiver) = channel(4);
        sender.send(output(&[]));
        let mut progress = output(&[]);
        progress.output.prefill_progress = Some(Box::new(PrefillProgress {
            request_id: "req".to_owned(),
            sequence: 1,
            phase: 3,
            prompt_tokens: 100,
            total_compute_tokens: 100,
            completed_compute_tokens: 50,
            remaining_compute_tokens: 50,
            local_cached_tokens: 0,
            external_cached_tokens: 0,
            pending_kv_transfer_tokens: 0,
            cache_accounting_final: true,
        }));
        sender.send(progress.clone());

        receiver.try_recv().unwrap().unwrap();
        let delivery = receiver.try_recv().unwrap().unwrap();
        assert_eq!(
            delivery.last().prefill_progress.as_ref().unwrap().remaining_compute_tokens,
            50
        );
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));

        let (mut sender, mut receiver) = channel(4);
        sender.send(progress);
        sender.send(output(&[1]));
        assert_eq!(deliveries(&mut receiver), vec![vec![vec![]], vec![vec![1]]]);
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
        while let Ok(item) = receiver.try_recv() {
            items.push(item.map(|delivery| token_ids(delivery).len()).ok());
        }
        // One delivery for the first output, one for the held-back output,
        // then the error.
        assert_eq!(items, [Some(1), Some(1), None]);
    }
}
