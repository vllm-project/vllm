// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `packed_top_k_logprobs` encoding of raw generate output logprobs (the
//! `PackedTopK` layout). Every engine step is encoded as it arrives, and the
//! encoded segments are sent in the response body without being copied into
//! one buffer.

use std::convert::Infallible;
use std::io::Write as _;

use axum::body::Body;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use base64::engine::GeneralPurpose;
use base64::engine::general_purpose::STANDARD;
use base64::write::EncoderWriter;
use bytes::Bytes;
use vllm_engine_core_client::protocol::logprobs::Logprobs;

use super::types::{GenerateLogProbs, GenerateResponse, PackedTopK};
use crate::error::ApiError;

/// Upper bound on one encoded segment handed to the HTTP body.
const TARGET_BYTES: usize = 1 << 20;

/// Encoded output kept as immutable segments of at most `TARGET_BYTES`, so
/// the response body can send them without copying.
#[derive(Default)]
struct Segments {
    done: Vec<Bytes>,
    current: Vec<u8>,
}

impl std::io::Write for Segments {
    fn write(&mut self, mut buf: &[u8]) -> std::io::Result<usize> {
        // Always accepts all of `buf`: `EncoderWriter` reports a partial
        // write as `Ok(0)`, which `write_all` treats as an error.
        let len = buf.len();
        while !buf.is_empty() {
            if self.current.len() == TARGET_BYTES {
                self.done.push(Bytes::from(std::mem::take(&mut self.current)));
            }
            if self.current.capacity() == 0 {
                // One allocation per segment (no growth copies or freed fragments).
                self.current.reserve_exact(TARGET_BYTES);
            }
            let (now, rest) = buf.split_at(buf.len().min(TARGET_BYTES - self.current.len()));
            self.current.extend_from_slice(now);
            buf = rest;
        }
        Ok(len)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

type Encoder = EncoderWriter<'static, GeneralPurpose, Segments>;

fn finish_encoder(mut encoder: Encoder) -> Vec<Bytes> {
    let mut segments = encoder.finish().expect("encoding into memory cannot fail");
    if !segments.current.is_empty() {
        segments.current.shrink_to_fit();
        segments.done.push(Bytes::from(segments.current));
    }
    segments.done
}

/// Accumulates sample logprobs directly into the `top_k` encoding.
pub(super) struct TopKLogprobsAccumulator {
    /// Engine row width `k + 1`. Wider (padded) rows are truncated to it;
    /// narrower rows are an error.
    width: usize,
    num_positions: usize,
    token_ids: Encoder,
    logprobs: Encoder,
    row: Vec<u8>,
    error: Option<String>,
}

impl TopKLogprobsAccumulator {
    /// `k` is the requested `sampling_params.logprobs`.
    pub(super) fn new(k: usize) -> Self {
        Self {
            width: k + 1,
            num_positions: 0,
            token_ids: EncoderWriter::new(Segments::default(), &STANDARD),
            logprobs: EncoderWriter::new(Segments::default(), &STANDARD),
            row: Vec::new(),
            error: None,
        }
    }

    /// Consume one engine output: `new_tokens` generated tokens and their
    /// logprobs, which must hold exactly one position per token. The first
    /// malformed output fails the whole block.
    pub(super) fn push(&mut self, new_tokens: usize, logprobs: Option<Logprobs>) {
        if self.error.is_none()
            && let Err(error) = self.try_push(new_tokens, logprobs)
        {
            self.error = Some(error);
        }
    }

    fn try_push(&mut self, new_tokens: usize, logprobs: Option<Logprobs>) -> Result<(), String> {
        let positions = logprobs.as_ref().map_or(0, Logprobs::len);
        if positions != new_tokens {
            return Err(format!(
                "raw generate output carried {positions} logprob positions for {new_tokens} new tokens"
            ));
        }
        let Some(logprobs) = logprobs else {
            return Ok(());
        };
        let rows = &logprobs.positions;
        if let Some(narrow) = rows.iter().find(|p| p.entries.len() < self.width) {
            return Err(format!(
                "raw generate logprobs row has {} candidates, expected at least {}",
                narrow.entries.len(),
                self.width
            ));
        }
        let top_k = || rows.iter().flat_map(|p| &p.entries[1..self.width]);
        if let Some(id) = top_k().map(|e| e.token_id).find(|&id| i32::try_from(id).is_err()) {
            return Err(format!(
                "raw generate logprobs token id {id} does not fit int32"
            ));
        }
        self.row.clear();
        self.row.extend(top_k().flat_map(|e| e.token_id.to_le_bytes()));
        self.token_ids.write_all(&self.row).expect("encoding into memory cannot fail");
        self.row.clear();
        self.row.extend(top_k().flat_map(|e| e.logprob.to_le_bytes()));
        self.logprobs.write_all(&self.row).expect("encoding into memory cannot fail");
        self.num_positions += positions;
        Ok(())
    }

    /// Finish encoding and return the logprobs of the choice.
    pub(super) fn finish(self) -> Result<TopKLogprobs, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        Ok(TopKLogprobs {
            num_positions: self.num_positions,
            k: self.width - 1,
            token_ids: finish_encoder(self.token_ids),
            logprobs: finish_encoder(self.logprobs),
        })
    }
}

/// Finished `packed_top_k_logprobs` output for one choice, with the arrays as
/// base64 segments.
pub(super) struct TopKLogprobs {
    pub num_positions: usize,
    pub k: usize,
    pub token_ids: Vec<Bytes>,
    pub logprobs: Vec<Bytes>,
}

/// Build the HTTP response for a non-streaming generate request whose single
/// choice carries `logprobs` as the `top_k` block.
///
/// The response is serialized by serde with a random placeholder for each
/// encoded array; the body then sends the encoded segments in place of the
/// placeholders, with an exact `Content-Length`.
pub(super) fn top_k_response(mut response: GenerateResponse, logprobs: TopKLogprobs) -> Response {
    let tag = uuid::Uuid::new_v4().simple();
    let ids_placeholder = format!("vllm-top-k-token-ids-{tag}");
    let logprobs_placeholder = format!("vllm-top-k-logprobs-{tag}");
    response.choices[0].logprobs = Some(GenerateLogProbs {
        top_k: Some(PackedTopK {
            num_positions: logprobs.num_positions,
            k: logprobs.k,
            token_ids: ids_placeholder.clone(),
            logprobs: logprobs_placeholder.clone(),
        }),
        ..Default::default()
    });
    let Ok(body) = serde_json::to_vec(&response) else {
        return serialize_error();
    };
    let body = Bytes::from(body);
    // The placeholders are unguessable, so each occurs only in its field;
    // `token_ids` serializes first.
    let find = |needle: &str, from: usize| {
        (body[from..]
            .windows(needle.len())
            .position(|window| window == needle.as_bytes()))
        .map(|offset| from + offset)
    };
    let Some(ids_at) = find(&ids_placeholder, 0) else {
        return serialize_error();
    };
    let Some(logprobs_at) = find(&logprobs_placeholder, ids_at) else {
        return serialize_error();
    };
    let mut parts = vec![body.slice(..ids_at)];
    parts.extend(logprobs.token_ids);
    parts.push(body.slice(ids_at + ids_placeholder.len()..logprobs_at));
    parts.extend(logprobs.logprobs);
    parts.push(body.slice(logprobs_at + logprobs_placeholder.len()..));

    let content_length: usize = parts.iter().map(Bytes::len).sum();
    let parts = parts.into_iter().map(Ok::<_, Infallible>);
    let mut response = Body::from_stream(futures::stream::iter(parts)).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(content_length));
    response
}

fn serialize_error() -> Response {
    ApiError::server_error("failed to serialize raw generate response".to_string()).into_response()
}

#[cfg(test)]
pub(super) mod tests {
    use base64::Engine as _;
    use vllm_engine_core_client::protocol::logprobs::PositionLogprobs;

    use super::super::tests::position;
    use super::*;

    /// Decoded `top_k` arrays: (token_ids, logprob bits).
    pub(crate) fn decode_top_k(block: &serde_json::Value) -> (Vec<i32>, Vec<u32>) {
        let words = |key: &str| {
            let bytes = STANDARD
                .decode(block[key].as_str().expect("base64 string"))
                .expect("valid base64");
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let ids = words("token_ids").into_iter().map(|w| w as i32).collect();
        (ids, words("logprobs"))
    }

    /// Engine slots `1..=k` of each row: (ids, logprob bits).
    pub(crate) fn engine_top_k(rows: &[PositionLogprobs], k: usize) -> (Vec<i32>, Vec<u32>) {
        let slots = || rows.iter().flat_map(|p| &p.entries[1..=k]);
        (
            slots().map(|e| e.token_id as i32).collect(),
            slots().map(|e| e.logprob.to_bits()).collect(),
        )
    }

    fn push_rows(accumulator: &mut TopKLogprobsAccumulator, rows: Vec<PositionLogprobs>) {
        accumulator.push(rows.len(), Some(Logprobs { positions: rows }));
    }

    #[test]
    fn segments_are_bounded_and_concatenate_to_one_shot_encoding() {
        let data: Vec<u8> = (0..7_000_003_u32).map(|i| (i * 31 + 7) as u8).collect();
        let mut encoder = EncoderWriter::new(Segments::default(), &STANDARD);
        let mut offset = 0;
        // Tiny writes (carry across groups) and writes several segments long.
        for size in [1_usize, 2, 300_000, 5, 3_000_001, 700_000, 1].iter().cycle() {
            if offset >= data.len() {
                break;
            }
            let end = (offset + size).min(data.len());
            encoder.write_all(&data[offset..end]).unwrap();
            offset = end;
        }
        let segments = finish_encoder(encoder);
        assert!(segments.iter().all(|s| !s.is_empty() && s.len() <= TARGET_BYTES));
        assert!(segments.concat() == STANDARD.encode(&data).into_bytes());
    }

    #[test]
    fn top_k_round_trip_preserves_engine_rows_and_raw_bits() {
        let nan_payload = f32::from_bits(0x7fc0_1234);
        // The first row is padded to a wider batch-wide top-k and truncated.
        let rows = [
            position(&[
                (5, -0.25, 3),
                (9, -0.1, 1),
                (7, -0.2, 2),
                (5, -0.25, 3),
                (4, -9.0, 9),
            ]),
            position(&[
                (11, f32::NEG_INFINITY, 40),
                (1, -0.0, 1),
                (2, f32::NEG_INFINITY, 2),
                (3, nan_payload, 3),
            ]),
        ];
        let mut accumulator = TopKLogprobsAccumulator::new(3);
        push_rows(&mut accumulator, rows[..1].to_vec());
        push_rows(&mut accumulator, rows[1..].to_vec());
        let logprobs = accumulator.finish().unwrap();
        assert_eq!((logprobs.num_positions, logprobs.k), (2, 3));
        let value = serde_json::json!({
            "token_ids": String::from_utf8(logprobs.token_ids.concat()).unwrap(),
            "logprobs": String::from_utf8(logprobs.logprobs.concat()).unwrap(),
        });
        let (token_ids, bits) = decode_top_k(&value);
        assert_eq!((token_ids.clone(), bits.clone()), engine_top_k(&rows, 3));
        // Top-k slots only, in engine order.
        assert_eq!(token_ids, vec![9, 7, 5, 1, 2, 3]);
        // No clamp: -0.0 sign, -inf and the NaN payload are preserved.
        assert_eq!(bits[3], (-0.0_f32).to_bits());
        assert_eq!(f32::from_bits(bits[4]), f32::NEG_INFINITY);
        assert_eq!(bits[5], 0x7fc0_1234);
    }
}
