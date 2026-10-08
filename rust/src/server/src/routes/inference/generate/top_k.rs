// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `return_top_k_logprobs` encoding of raw generate output logprobs.
//!
//! `top_k` carries the engine top-k candidates of every position (slots `1..=k`
//! of the engine row, in engine order): token ids as little-endian `int32`,
//! logprobs as the raw engine `float32` bits (no `-9999` clamp), each array as
//! standard padded base64, row-major `N x k`.
//! With `return_token_logprobs`, `sampled` carries the sampled token's logprob
//! per position (engine slot 0, clamped like `content`). Every engine step is
//! encoded as it arrives.

use std::io::Write as _;

use base64::engine::GeneralPurpose;
use base64::engine::general_purpose::STANDARD;
use base64::write::EncoderWriter;
use bytes::Bytes;
use vllm_engine_core_client::protocol::logprobs::Logprobs;

use crate::routes::openai::utils::logprobs::clamp_logprob;

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
    sampled: Option<Vec<f64>>,
    token_ids: Encoder,
    logprobs: Encoder,
    row: Vec<u8>,
    error: Option<String>,
}

impl TopKLogprobsAccumulator {
    /// `k` is the requested `sampling_params.logprobs`; `sampled` is
    /// `return_token_logprobs`.
    pub(super) fn new(k: usize, sampled: bool) -> Self {
        Self {
            width: k + 1,
            num_positions: 0,
            sampled: sampled.then(Vec::new),
            token_ids: EncoderWriter::new(Segments::default(), &STANDARD),
            logprobs: EncoderWriter::new(Segments::default(), &STANDARD),
            row: Vec::new(),
            error: None,
        }
    }

    /// Consume one engine output: `new_tokens` generated tokens and their
    /// logprobs, which must hold exactly one position per token.
    pub(super) fn push(&mut self, new_tokens: usize, logprobs: Option<Logprobs>) {
        if self.error.is_some() {
            return;
        }
        let positions = logprobs.as_ref().map_or(0, Logprobs::len);
        if positions != new_tokens {
            self.error = Some(format!(
                "raw generate output carried {positions} logprob positions for {new_tokens} new tokens"
            ));
            return;
        }
        let Some(logprobs) = logprobs else { return };
        if let Some(narrow) = logprobs.positions.iter().find(|p| p.entries.len() < self.width) {
            self.error = Some(format!(
                "raw generate logprobs row has {} candidates, expected at least {}",
                narrow.entries.len(),
                self.width
            ));
            return;
        }
        let top_k_ids = logprobs.positions.iter().flat_map(|p| &p.entries[1..self.width]);
        if let Some(id) = top_k_ids.map(|e| e.token_id).find(|&id| i32::try_from(id).is_err()) {
            self.error = Some(format!(
                "raw generate logprobs token id {id} does not fit int32"
            ));
            return;
        }
        for position in &logprobs.positions {
            if let Some(sampled) = self.sampled.as_mut() {
                sampled.push(f64::from(clamp_logprob(position.entries[0].logprob)));
            }
            let top_k = &position.entries[1..self.width];
            self.row.clear();
            self.row.extend(top_k.iter().flat_map(|e| e.token_id.to_le_bytes()));
            self.token_ids.write_all(&self.row).expect("encoding into memory cannot fail");
            self.row.clear();
            self.row.extend(top_k.iter().flat_map(|e| e.logprob.to_bits().to_le_bytes()));
            self.logprobs.write_all(&self.row).expect("encoding into memory cannot fail");
        }
        self.num_positions += positions;
    }

    /// Finish encoding and return the logprobs of the choice.
    pub(super) fn finish(self) -> Result<TopKLogprobs, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        Ok(TopKLogprobs {
            sampled: self.sampled,
            top_k: TopKBlock {
                num_positions: self.num_positions,
                k: self.width - 1,
                token_ids: finish_encoder(self.token_ids),
                logprobs: finish_encoder(self.logprobs),
            },
        })
    }
}

/// Finished `return_top_k_logprobs` output for one choice.
pub(super) struct TopKLogprobs {
    /// Present with `return_token_logprobs`.
    pub sampled: Option<Vec<f64>>,
    pub top_k: TopKBlock,
}

/// Top-k candidates (arrays as base64 segments).
pub(super) struct TopKBlock {
    pub num_positions: usize,
    pub k: usize,
    pub token_ids: Vec<Bytes>,
    pub logprobs: Vec<Bytes>,
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

    fn finish_top_k(accumulator: TopKLogprobsAccumulator) -> TopKBlock {
        accumulator.finish().unwrap().top_k
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
    fn top_k_truncates_padded_rows_to_requested_k() {
        // Engine rows padded to a wider batch-wide top-k (4 slots) for a
        // request that asked for k = 1.
        let mut accumulator = TopKLogprobsAccumulator::new(1, false);
        push_rows(
            &mut accumulator,
            vec![
                position(&[(5, -0.5, 2), (4, -0.25, 1), (5, -0.5, 2), (6, -1.0, 3)]),
                position(&[(7, -0.1, 1), (7, -0.1, 1), (8, -2.0, 2), (9, -3.0, 3)]),
            ],
        );
        let top_k = finish_top_k(accumulator);
        assert_eq!((top_k.num_positions, top_k.k), (2, 1));
        assert_eq!(
            top_k.token_ids.concat(),
            STANDARD.encode([4_u8, 0, 0, 0, 7, 0, 0, 0]).as_bytes()
        );
    }

    #[test]
    fn top_k_round_trip_preserves_engine_rows_and_raw_bits() {
        let nan_payload = f32::from_bits(0x7fc0_1234);
        let rows = [
            position(&[(5, -0.25, 3), (9, -0.1, 1), (7, -0.2, 2), (5, -0.25, 3)]),
            position(&[
                (11, f32::NEG_INFINITY, 40),
                (1, -0.0, 1),
                (2, f32::NEG_INFINITY, 2),
                (3, nan_payload, 3),
            ]),
        ];
        let mut accumulator = TopKLogprobsAccumulator::new(3, true);
        push_rows(&mut accumulator, rows[..1].to_vec());
        push_rows(&mut accumulator, rows[1..].to_vec());
        let logprobs = accumulator.finish().unwrap();
        // The sampled slot is clamped like `content`.
        assert_eq!(logprobs.sampled, Some(vec![-0.25, -9999.0]));
        let top_k = logprobs.top_k;
        assert_eq!((top_k.num_positions, top_k.k), (2, 3));
        let value = serde_json::json!({
            "token_ids": String::from_utf8(top_k.token_ids.concat()).unwrap(),
            "logprobs": String::from_utf8(top_k.logprobs.concat()).unwrap(),
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

    #[test]
    fn top_k_rejects_emitted_ids_beyond_int32() {
        // Only emitted (top-k) ids are checked; the sampled slot and padding are not.
        let finish = |id: u32| {
            let mut accumulator = TopKLogprobsAccumulator::new(1, false);
            push_rows(
                &mut accumulator,
                vec![position(&[
                    (u32::MAX, -0.5, 1),
                    (id, -0.5, 1),
                    (u32::MAX, -1.0, 2),
                ])],
            );
            accumulator.finish().map(|logprobs| logprobs.top_k.num_positions)
        };
        assert_eq!(finish(i32::MAX as u32), Ok(1));
        assert!(finish(i32::MAX as u32 + 1).is_err());
        assert!(finish(u32::MAX).is_err());
    }

    #[test]
    fn top_k_rejects_rows_narrower_than_requested_k() {
        // k = 2 -> width 3 is fixed by the request; a narrow first row fails
        // the block instead of shrinking it.
        let mut accumulator = TopKLogprobsAccumulator::new(2, false);
        push_rows(
            &mut accumulator,
            vec![position(&[(1, -0.1, 1), (1, -0.1, 1)])],
        );
        push_rows(
            &mut accumulator,
            vec![position(&[(2, -0.1, 1), (2, -0.1, 1), (3, -0.2, 2)])],
        );
        assert!(accumulator.finish().is_err());
    }
}
