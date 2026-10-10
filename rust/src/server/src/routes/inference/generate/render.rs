// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Direct JSON rendering for non-streaming raw generate responses.
//!
//! The non-streaming response can hold hundreds of thousands of positions with
//! 100+ candidates each. Instead of building a `GenerateLogProbs` tree and
//! serializing it into one contiguous buffer, the output logprobs are written
//! straight from the engine logprobs in bounded chunks by a task on the runtime
//! that built the response (the request runtime for the offloaded generate
//! route) into a bounded channel that the HTTP body drains. The produced bytes
//! are identical to serializing the `GenerateResponse` carrying the
//! `GenerateLogProbs` built by `raw_logprobs_to_generate` with `serde_json`.

use std::collections::HashSet;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::Json;
use axum::body::Body;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http_body::Frame;
use serde::Serialize;
use serde_json::value::RawValue;
use tokio::sync::mpsc;
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::logprobs::{PositionLogprobs, TokenLogprob};

use super::types::GenerateResponse;
use super::{top_logprob_entries, wire_rank};
use crate::error::ApiError;
use crate::routes::openai::utils::logprobs::clamp_logprob;

/// Output logprobs of the single choice, rendered by [`generate_response`].
pub(super) enum ChoiceLogprobs {
    /// `"logprobs": null`.
    None,
    /// `{"content": [...]}` for `sampling_params.logprobs = requested`.
    /// Positions must be non-empty.
    Generate {
        positions: Vec<PositionLogprobs>,
        requested: i32,
    },
}

/// Rendered output logprobs per body chunk (~160 positions at top-128).
const BODY_CHUNK_BYTES: usize = 1 << 20;

/// Chunks rendered for an in-flight body, at most this many buffered ahead of
/// the socket.
const BODY_CHANNEL_CHUNKS: usize = 2;

/// Build the HTTP response for one non-streaming generate request whose choice
/// carries no `logprobs` (they are passed separately).
pub(super) fn generate_response(response: GenerateResponse, logprobs: ChoiceLogprobs) -> Response {
    let (positions, requested) = match logprobs {
        ChoiceLogprobs::None => return Json(response).into_response(),
        ChoiceLogprobs::Generate {
            positions,
            requested,
        } => (positions, requested),
    };
    let Some((mut head, choice_tail)) = split_response(response) else {
        return ApiError::server_error("failed to serialize raw generate response".to_string())
            .into_response();
    };
    head.extend_from_slice(b"{\"content\":[");
    let mut tail = b"]}".to_vec();
    tail.extend_from_slice(&choice_tail);

    let (tx, rx) = mpsc::channel(BODY_CHANNEL_CHUNKS);
    tokio::spawn(
        produce_body(head.into(), positions, requested, tail.into(), tx)
            .instrument(tracing::Span::current()),
    );
    let mut response = Body::new(ChannelBody {
        rx,
        complete: false,
    })
    .into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Serialize `response` around the first choice's `logprobs` value: the bytes
/// before the value and the bytes after it. The value is serialized as a random
/// placeholder, which no client-supplied string can reproduce.
fn split_response(mut response: GenerateResponse) -> Option<(Vec<u8>, Vec<u8>)> {
    let placeholder = format!("\"vllm-logprobs-{}\"", uuid::Uuid::new_v4());
    response.choices[0].logprobs = Some(RawValue::from_string(placeholder.clone()).ok()?);
    let mut head = serde_json::to_vec(&response).ok()?;
    let mut matches = head
        .windows(placeholder.len())
        .enumerate()
        .filter(|(_, window)| *window == placeholder.as_bytes());
    let (Some((start, _)), None) = (matches.next(), matches.next()) else {
        return None;
    };
    let tail = head.split_off(start + placeholder.len());
    head.truncate(start);
    Some((head, tail))
}

/// Render one body (head, output logprobs, tail) into `tx`, releasing each
/// position after it is written.
///
/// Runs as a task on the runtime that built the response, which for the
/// offloaded generate route is the request runtime, so the HTTP runtime only
/// moves ready chunks. `send` waits while the channel is full (backpressure
/// from the client); a dropped body closes the channel and stops rendering.
/// `None` marks the end of the body; an empty row ends the task without it.
async fn produce_body(
    head: Bytes,
    positions: Vec<PositionLogprobs>,
    requested: i32,
    tail: Bytes,
    tx: mpsc::Sender<Option<Bytes>>,
) {
    if tx.send(Some(head)).await.is_err() {
        return;
    }
    let mut seen = HashSet::new();
    let mut positions = positions.into_iter();
    let mut first = true;
    while positions.len() > 0 {
        // Slack for the position that crosses the limit.
        let mut chunk = Vec::with_capacity(BODY_CHUNK_BYTES + (64 << 10));
        while chunk.len() < BODY_CHUNK_BYTES
            && let Some(position) = positions.next()
        {
            if !std::mem::replace(&mut first, false) {
                chunk.push(b',');
            }
            if write_position(&mut chunk, &position, requested, &mut seen).is_none() {
                return;
            }
            #[cfg(test)]
            tests::RENDERED_ON_THREAD.with(|count| count.set(count.get() + 1));
        }
        if tx.send(Some(Bytes::from(chunk))).await.is_err() {
            return;
        }
        // Keep the request runtime fair when the client drains faster than
        // we render (sends then never wait).
        tokio::task::yield_now().await;
    }
    if tx.send(Some(tail)).await.is_err() {
        return;
    }
    let _ = tx.send(None).await;
}

/// HTTP body fed by [`produce_body`]; size unknown (chunked).
///
/// If the producer stops without its completion marker (cancelled, or an
/// empty row), the body yields an error so hyper aborts the connection instead
/// of writing the final chunk terminator after truncated JSON.
struct ChannelBody {
    rx: mpsc::Receiver<Option<Bytes>>,
    complete: bool,
}

impl http_body::Body for ChannelBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
        if self.complete {
            return Poll::Ready(None);
        }
        self.rx.poll_recv(cx).map(|message| match message {
            Some(Some(chunk)) => Some(Ok(Frame::data(chunk))),
            Some(None) => {
                self.complete = true;
                None
            }
            None => Some(Err(std::io::Error::other(
                "generate response body render task ended before completion",
            ))),
        })
    }

    fn is_end_stream(&self) -> bool {
        self.complete
    }
}

/// Write one `GenerateLogProbsContent` object, byte-identical to `serde_json`
/// serialization of `position_to_generate_logprobs_content(position,
/// requested)`. Writes nothing and returns `None` if `position.entries` is
/// empty.
fn write_position(
    out: &mut Vec<u8>,
    position: &PositionLogprobs,
    requested: i32,
    seen: &mut HashSet<u32>,
) -> Option<()> {
    write_candidate_fields(out, position.entries.first()?);
    out.extend_from_slice(b",\"top_logprobs\":[");
    for (index, entry) in top_logprob_entries(position, requested, seen).enumerate() {
        if index > 0 {
            out.push(b',');
        }
        write_candidate_fields(out, entry);
        out.push(b'}');
    }
    out.extend_from_slice(b"]}");
    Some(())
}

/// Write `{"token_id":N,"logprob":X,"rank":R` (no closing brace).
fn write_candidate_fields(out: &mut Vec<u8>, entry: &TokenLogprob) {
    out.extend_from_slice(b"{\"token_id\":");
    json(out, &entry.token_id);
    out.extend_from_slice(b",\"logprob\":");
    json(out, &clamp_logprob(entry.logprob));
    out.extend_from_slice(b",\"rank\":");
    json(out, &wire_rank(entry.rank));
}

/// Only used for values whose serialization is total (strings, numbers,
/// options and vectors of them, maps with `u32` or string keys,
/// `serde_json::Value`) into a `Vec<u8>`, which cannot fail; `axum::Json` on
/// the same response has the same failure modes.
fn json<T: Serialize + ?Sized>(out: &mut Vec<u8>, value: &T) {
    serde_json::to_writer(out, value).expect("generate response must serialize");
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::super::types::GenerateResponseChoice;
    use super::*;

    thread_local! {
        /// Positions rendered on the current thread.
        pub(super) static RENDERED_ON_THREAD: std::cell::Cell<usize> =
            const { std::cell::Cell::new(0) };
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn render_task_failure_errors_the_body_instead_of_ending_it() {
        // An empty row slipping past validation stops the render task midway
        // (without panicking, which aborts release builds). The body must fail,
        // so hyper aborts the connection rather than writing a clean chunked
        // terminator after truncated JSON.
        let mut rows = positions(300, 3);
        rows[200].entries.clear();
        let response = generate_response(test_response(), generate(rows));
        let result = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        assert!(result.is_err(), "truncated body completed cleanly");
    }

    /// Deterministic xorshift for test data.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn direct_writer_matches_serde_reference() {
        let specials = [
            0.0_f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -9999.0,
            -9999.5,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE / 3.0,
            f32::MAX,
            f32::MIN,
            -1e-7,
            -0.1,
        ];
        let requested = [-1, 0, 1, 2, 5, 8];
        let mut seen = HashSet::new();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for row in 0..3000 {
            let width = 1 + (next(&mut state) % 8) as usize;
            let entries = (0..width)
                .map(|_| {
                    // Small ids repeat within a row.
                    let token_id = match next(&mut state) % 4 {
                        0 => (next(&mut state) % 10) as u32,
                        1 => u32::MAX - (next(&mut state) % 2) as u32,
                        _ => (next(&mut state) % 200_000) as u32,
                    };
                    let logprob = if next(&mut state).is_multiple_of(4) {
                        specials[(next(&mut state) % specials.len() as u64) as usize]
                    } else {
                        f32::from_bits(next(&mut state) as u32)
                    };
                    TokenLogprob {
                        token_id,
                        logprob,
                        // Rank 0 (NaN sampled logprob) goes on the wire as null.
                        rank: [0, 1, 77, u32::MAX][(next(&mut state) % 4) as usize],
                    }
                })
                .collect();
            let position = PositionLogprobs { entries };
            let requested = requested[(next(&mut state) % requested.len() as u64) as usize];
            let expected = serde_json::to_vec(
                &super::super::position_to_generate_logprobs_content(&position, requested).unwrap(),
            )
            .unwrap();
            let mut direct = Vec::new();
            write_position(&mut direct, &position, requested, &mut seen).unwrap();
            assert_eq!(
                String::from_utf8_lossy(&direct),
                String::from_utf8_lossy(&expected),
                "row {row}"
            );
        }
    }

    #[tokio::test]
    async fn producer_stops_when_body_is_dropped_and_is_bounded() {
        // ~6 KB per position: several chunks.
        let many = positions(1000, 129);
        let (tx, mut rx) = mpsc::channel(BODY_CHANNEL_CHUNKS);
        let producer = tokio::spawn(produce_body(
            Bytes::from_static(b"["),
            many,
            128,
            Bytes::from_static(b"]"),
            tx,
        ));
        // Head + one rendered chunk, then let the producer fill the channel.
        assert_eq!(rx.recv().await, Some(Some(Bytes::from_static(b"["))));
        rx.recv().await.unwrap();
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        // Backpressure: the producer waits on the full channel.
        assert!(!producer.is_finished());
        drop(rx);
        tokio::time::timeout(std::time::Duration::from_secs(5), producer)
            .await
            .expect("producer stops after the body is dropped")
            .unwrap();
    }

    #[test]
    fn render_runs_on_request_runtime_not_body_poller() {
        // `generate_response` runs inside the handler on the request runtime;
        // the HTTP runtime only polls the body. Rendering must stay on the
        // former so control routes on the HTTP runtime are not starved.
        let request_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let response = request_runtime
            .block_on(async { generate_response(test_response(), generate(positions(300, 3))) });
        let http_runtime =
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        RENDERED_ON_THREAD.with(|count| count.set(0));
        let body = http_runtime
            .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
            .unwrap();
        assert_eq!(
            RENDERED_ON_THREAD.with(|count| count.get()),
            0,
            "positions were rendered on the body-polling thread"
        );
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["choices"][0]["logprobs"]["content"].as_array().unwrap().len(),
            300
        );
    }

    fn generate(positions: Vec<PositionLogprobs>) -> ChoiceLogprobs {
        ChoiceLogprobs::Generate {
            positions,
            requested: 2,
        }
    }

    fn test_response() -> GenerateResponse {
        GenerateResponse {
            request_id: "http-1".to_string(),
            choices: vec![GenerateResponseChoice {
                index: 0,
                logprobs: None,
                finish_reason: Some("abort".to_string()),
                token_ids: vec![1, 2],
                sampling_mask: None,
            }],
            prompt_logprobs: None,
            prompt_token_id_logprobs: None,
            prompt_token_ids: None,
            mm_placeholders: None,
            kv_transfer_params: None,
            ec_transfer_params: None,
            metrics: None,
        }
    }

    fn positions(count: u32, width: u32) -> Vec<PositionLogprobs> {
        (0..count)
            .map(|i| PositionLogprobs {
                entries: (0..width)
                    .map(|j| TokenLogprob {
                        token_id: i + j,
                        logprob: -(j as f32) * 0.5,
                        rank: j + 1,
                    })
                    .collect(),
            })
            .collect()
    }
}
