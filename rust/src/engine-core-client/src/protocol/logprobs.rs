// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub(crate) mod array;
#[cfg(test)]
mod tests;
mod wire;

use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize};

use self::array::DecodedRanks;
use self::wire::*;
use crate::error::{Error, Result, bail_ext_value_decode};
use crate::protocol::dtype::{NumpyDtype, TensorDtype};
use crate::protocol::tensor::{WireArrayData, WireNdArray};

/// One token candidate and its logprob metadata for a single sequence position.
///
/// The first entry in a [`PositionLogprobs`] is always the sampled/selected
/// token for that position. Any remaining entries follow the engine's returned
/// candidate order: the top-k, or the requested `logprob_token_ids`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    /// Preserves the engine's value, including NaN and infinities.
    pub logprob: f32,
    /// The sampled/selected token uses its actual vocab rank. Remaining entries
    /// use 1-based top-k ranks matching the engine's returned candidate
    /// order, unless the engine sends a rank per entry (`logprob_token_ids`),
    /// in which case every entry uses its actual vocab rank.
    /// A sampled/selected rank of 0 occurs when its logprob is NaN: the engine's
    /// `(logprobs >= selected_logprob).sum(-1)` counts no matching values.
    pub rank: u32,
}

/// Logprob payload for one sequence position.
///
/// This is the semantic Rust representation used by the public client API after
/// the lower-level ndarray/tensor wire payload has been decoded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionLogprobs {
    pub entries: Vec<TokenLogprob>,
}

impl PositionLogprobs {
    /// Convert one decoded logprobs row into this per-position form by grouping
    /// each token/logprob pair together with its rank.
    fn from_decoded_row(
        token_ids: &[u32],
        logprobs: &[f32],
        ranks: impl IntoIterator<Item = u32>,
    ) -> Result<Self> {
        if token_ids.len() != logprobs.len() {
            bail_ext_value_decode!(
                "logprobs row length mismatch: token_ids={}, logprobs={}",
                token_ids.len(),
                logprobs.len()
            );
        }
        let entries = token_ids
            .iter()
            .zip(logprobs)
            .zip(ranks)
            .map(|((&token_id, &logprob), rank)| TokenLogprob {
                token_id,
                logprob,
                rank,
            })
            .collect();
        Ok(Self { entries })
    }
}

/// Decoded per-request logprobs payload for one engine-core output.
///
/// Unlike the Python wire payload, this public Rust type is already fully
/// semantic: one [`PositionLogprobs`] per scored position, each containing the
/// sampled/selected token plus any returned top-k alternatives for that same
/// position.
///
/// The Python engine still sends logprobs as ndarray/tensor-shaped wire tuples.
/// Rust resolves that lower-level representation during decode and exposes only
/// this per-position form to callers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Logprobs {
    /// One decoded logprobs record per scored position in this engine-core
    /// output.
    pub positions: Vec<PositionLogprobs>,
}

impl Logprobs {
    /// Returns the number of scored positions in this payload.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns whether the payload contains no scored positions.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Output field wrapper that is initially deserialized from the Python wire
/// shape, then resolved into [`Logprobs`] before the decoded message is
/// returned to callers.
#[derive(Clone, PartialEq, Debug, EnumAsInner)]
pub enum MaybeWireLogprobs {
    /// The logprobs are still in the wire format and need to be resolved by
    /// looking up aux frames and decoding raw views. Should only be used
    /// internally during deserialization.
    Wire(Box<WireLogprobs>),
    /// The actual decoded logprobs value,
    Direct(Logprobs),
}

impl Deref for MaybeWireLogprobs {
    type Target = Logprobs;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Wire(_) => panic!("Logprobs is still in wire format"),
            Self::Direct(value) => value,
        }
    }
}

impl DerefMut for MaybeWireLogprobs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Wire(_) => panic!("Logprobs is still in wire format"),
            Self::Direct(value) => value,
        }
    }
}

impl<'de> Deserialize<'de> for MaybeWireLogprobs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // When deserializing, it's always in the wire form.
        WireLogprobs::deserialize(deserializer).map(|v| Self::Wire(Box::new(v)))
    }
}

impl Serialize for MaybeWireLogprobs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // For testing purposes only. We don't actually serialize it into aux frames.
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(value) => WireLogprobs::from_direct(value)
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
        }
    }
}

impl MaybeWireLogprobs {
    /// Resolve the wire representation into decoded logprobs by looking up aux
    /// frames and decoding raw views as needed.
    pub(super) fn resolve(self, frames: &[Bytes], field_prefix: &str) -> Result<Self> {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => value.resolve(frames, field_prefix).map(Self::Direct),
        }
    }
}

impl WireLogprobs {
    /// Convert semantic per-position logprobs into the Python wire tuple shape.
    ///
    /// This exists mainly so Rust-side tests can inject semantic logprobs into
    /// mocked engine-core outputs without manually building ndarray
    /// raw-view tuples.
    fn from_direct(value: &Logprobs) -> std::result::Result<Self, String> {
        let rows = value.positions.len();
        let cols = value.positions.first().map(|position| position.entries.len()).unwrap_or(0);

        let mut token_ids = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(8));
        let mut logprobs = Vec::with_capacity(rows.saturating_mul(cols).saturating_mul(4));
        let mut token_ranks = Vec::with_capacity(rows.saturating_mul(8));

        for (row_index, position) in value.positions.iter().enumerate() {
            if position.entries.len() != cols {
                return Err(format!(
                    "logprobs row {row_index} length mismatch: expected {cols}, got {}",
                    position.entries.len()
                ));
            }
            let Some((sampled, _)) = position.entries.split_first() else {
                return Err(format!("logprobs row {row_index} is empty"));
            };

            token_ranks.extend_from_slice(&(sampled.rank as i64).to_le_bytes());
            for entry in &position.entries {
                token_ids.extend_from_slice(&(entry.token_id as i64).to_le_bytes());
                logprobs.extend_from_slice(&entry.logprob.to_le_bytes());
            }
        }

        Ok(Self {
            logprob_token_ids: WireNdArray {
                dtype: NumpyDtype::little(TensorDtype::I64),
                shape: vec![rows, cols],
                data: WireArrayData::RawView(token_ids.into()),
            },
            logprobs: WireNdArray {
                dtype: NumpyDtype::little(TensorDtype::F32),
                shape: vec![rows, cols],
                data: WireArrayData::RawView(logprobs.into()),
            },
            token_ranks: WireNdArray {
                dtype: NumpyDtype::little(TensorDtype::I64),
                shape: vec![rows],
                data: WireArrayData::RawView(token_ranks.into()),
            },
            cu_num_generated_tokens: None,
            cu_num_generated_tokens_tensor: None,
        })
    }

    /// Resolve the wire-format logprobs into semantic [`Logprobs`] records by
    /// looking up aux frames, decoding raw views, and grouping each row
    /// into one [`PositionLogprobs`].
    fn resolve(self, frames: &[Bytes], field_prefix: &str) -> Result<Logprobs> {
        if let Some(indices) = self.cu_num_generated_tokens {
            bail_ext_value_decode!(
                "{field_prefix}.cu_num_generated_tokens: \
                 expected None for per-request engine-core logprobs payload, got {indices:?}"
            );
        }

        // Unlike the sibling check above, don't Debug-print the payload:
        // an opaque non-None value here may embed a full tensor blob.
        if self.cu_num_generated_tokens_tensor.is_some() {
            bail_ext_value_decode!(
                "{field_prefix}.cu_num_generated_tokens_tensor: \
                 expected None for per-request engine-core logprobs payload"
            );
        }

        let token_ids = array::decode_array2_u32(
            self.logprob_token_ids,
            &format!("{field_prefix}.logprob_token_ids"),
            frames,
        )?;
        let logprobs =
            array::decode_array2_f32(self.logprobs, &format!("{field_prefix}.logprobs"), frames)?;
        let token_ranks = array::decode_ranks_u32(
            self.token_ranks,
            &format!("{field_prefix}.token_ranks"),
            frames,
        )?;

        if token_ids.rows != logprobs.rows || token_ids.cols != logprobs.cols {
            bail_ext_value_decode!(
                "{field_prefix}: row shape mismatch between token ids ({}, {}) and logprobs ({}, {})",
                token_ids.rows,
                token_ids.cols,
                logprobs.rows,
                logprobs.cols
            );
        }
        match &token_ranks {
            DecodedRanks::Sampled(ranks) if ranks.len() != token_ids.rows => {
                bail_ext_value_decode!(
                    "{field_prefix}: token_ranks length {} does not match row count {}",
                    ranks.len(),
                    token_ids.rows
                );
            }
            DecodedRanks::PerToken(ranks)
                if ranks.rows != token_ids.rows || ranks.cols != token_ids.cols =>
            {
                bail_ext_value_decode!(
                    "{field_prefix}: token_ranks shape ({}, {}) does not match token ids ({}, {})",
                    ranks.rows,
                    ranks.cols,
                    token_ids.rows,
                    token_ids.cols
                );
            }
            _ => {}
        }

        // Empty position lists may be encoded as either [0, 0] or [0, k + 1].
        if token_ids.rows == 0 {
            return Ok(Logprobs {
                positions: Vec::new(),
            });
        }
        if token_ids.cols == 0 {
            bail_ext_value_decode!(
                "{field_prefix}: zero-column logprobs payload with {} rows",
                token_ids.rows
            );
        }

        let rows = token_ids.data.chunks(token_ids.cols).zip(logprobs.data.chunks(logprobs.cols));
        let positions: Vec<PositionLogprobs> = match token_ranks {
            DecodedRanks::Sampled(ranks) => rows
                .zip(ranks)
                .map(|((token_ids_row, logprobs_row), sampled_rank)| {
                    PositionLogprobs::from_decoded_row(
                        token_ids_row,
                        logprobs_row,
                        std::iter::once(sampled_rank).chain(1..),
                    )
                })
                .collect::<Result<_>>()?,
            DecodedRanks::PerToken(ranks) => rows
                .zip(ranks.data.chunks(ranks.cols))
                .map(|((token_ids_row, logprobs_row), ranks_row)| {
                    PositionLogprobs::from_decoded_row(
                        token_ids_row,
                        logprobs_row,
                        ranks_row.iter().copied(),
                    )
                })
                .collect::<Result<_>>()?,
        };

        Ok(Logprobs { positions })
    }
}
