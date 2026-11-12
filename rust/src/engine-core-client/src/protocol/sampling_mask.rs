// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ops::{Deref, DerefMut};

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::error::{Error, Result, bail_ext_value_decode};
use crate::protocol::logprobs::array::{decode_array1_f32, decode_array1_u32};
use crate::protocol::tensor::WireNdArray;

/// Decoded sampling support and optional aligned logprobs, one row per token.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingMask {
    pub rows: Vec<Vec<u32>>,
    pub logprobs: Option<Vec<Vec<f32>>>,
}

impl SamplingMask {
    pub fn append(&mut self, mut other: Self) {
        self.rows.append(&mut other.rows);
        if let Some(mut scores) = other.logprobs {
            self.logprobs.get_or_insert_default().append(&mut scores);
        }
    }

    pub fn truncate(&mut self, len: usize) {
        self.rows.truncate(len);
        if let Some(scores) = &mut self.logprobs {
            scores.truncate(len);
        }
    }
}

/// Python `SamplingMaskLists` tuple before ndarray raw views are resolved.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireSamplingMask {
    token_ids: WireNdArray,
    #[serde(default)]
    offsets: Option<WireNdArray>,
    #[serde(default)]
    cu_num_generated_tokens: Option<Vec<usize>>,
    #[serde(default)]
    logprobs: Option<WireNdArray>,
}

/// Sampling-mask field while it transitions from Python wire data to rows.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum MaybeWireSamplingMask {
    Wire(Box<WireSamplingMask>),
    Direct(SamplingMask),
}

impl Deref for MaybeWireSamplingMask {
    type Target = SamplingMask;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Wire(_) => panic!("sampling mask is still in wire format"),
            Self::Direct(value) => value,
        }
    }
}

impl DerefMut for MaybeWireSamplingMask {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Wire(_) => panic!("sampling mask is still in wire format"),
            Self::Direct(value) => value,
        }
    }
}

impl<'de> Deserialize<'de> for MaybeWireSamplingMask {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireSamplingMask::deserialize(deserializer).map(|value| Self::Wire(Box::new(value)))
    }
}

impl Serialize for MaybeWireSamplingMask {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(value) => WireSamplingMask::from_direct(value)
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
        }
    }
}

impl MaybeWireSamplingMask {
    pub(super) fn resolve(self, frames: &[Bytes], field_prefix: &str) -> Result<Self> {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => value.resolve(frames, field_prefix).map(Self::Direct),
        }
    }
}

impl WireSamplingMask {
    fn from_direct(value: &SamplingMask) -> std::result::Result<Self, String> {
        let [row] = value.rows.as_slice() else {
            return Err(format!(
                "per-request engine-core sampling mask must contain exactly one row, got {}",
                value.rows.len()
            ));
        };
        if let Some(scores) = value.logprobs.as_ref()
            && (scores.len() != 1 || scores[0].len() != row.len())
        {
            return Err("sampling-mask logprobs must align with token IDs".into());
        }
        let token_ids = row.iter().map(|&token_id| i64::from(token_id)).collect::<Vec<_>>();
        Ok(Self {
            token_ids: WireNdArray::from_i64(vec![token_ids.len()], token_ids)?,
            offsets: None,
            cu_num_generated_tokens: None,
            logprobs: value
                .logprobs
                .as_ref()
                .map(|rows| WireNdArray::from_f32(vec![row.len()], rows[0].clone()))
                .transpose()?,
        })
    }

    fn resolve(self, frames: &[Bytes], field_prefix: &str) -> Result<SamplingMask> {
        if let Some(indices) = self.cu_num_generated_tokens {
            bail_ext_value_decode!(
                "{field_prefix}.cu_num_generated_tokens: expected None for per-request engine-core sampling-mask payload, got {indices:?}"
            );
        }

        let token_ids =
            decode_array1_u32(self.token_ids, &format!("{field_prefix}.token_ids"), frames)?;
        let logprobs = self
            .logprobs
            .map(|value| decode_array1_f32(value, &format!("{field_prefix}.logprobs"), frames))
            .transpose()?;
        if logprobs.as_ref().is_some_and(|values| values.len() != token_ids.len()) {
            bail_ext_value_decode!("{field_prefix}.logprobs: length must match token_ids");
        }
        let Some(offsets) = self.offsets else {
            return Ok(SamplingMask {
                rows: vec![token_ids],
                logprobs: logprobs.map(|values| vec![values]),
            });
        };
        let offsets = decode_array1_u32(offsets, &format!("{field_prefix}.offsets"), frames)?;
        if offsets.first().copied() != Some(0) {
            bail_ext_value_decode!("{field_prefix}.offsets: first offset must be zero");
        }
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            bail_ext_value_decode!("{field_prefix}.offsets: offsets must be nondecreasing");
        }
        if offsets.last().copied().map(|value| value as usize) != Some(token_ids.len()) {
            bail_ext_value_decode!(
                "{field_prefix}.offsets: final offset {:?} does not match token_ids length {}",
                offsets.last(),
                token_ids.len()
            );
        }

        Ok(SamplingMask {
            rows: split_rows(&token_ids, &offsets),
            logprobs: logprobs.map(|values| split_rows(&values, &offsets)),
        })
    }
}

/// Split CSR values into per-request rows using validated offsets.
fn split_rows<T: Clone>(values: &[T], offsets: &[u32]) -> Vec<Vec<T>> {
    offsets
        .windows(2)
        .map(|pair| values[pair[0] as usize..pair[1] as usize].to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dtype::{NumpyDtype, TensorDtype};
    use crate::protocol::output::{
        EngineCoreOutput, RequestBatchOutputs, decode_engine_core_outputs,
    };
    use crate::protocol::tensor::WireArrayData;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    #[test]
    fn decodes_single_position_without_offsets() {
        let wire = WireSamplingMask {
            token_ids: WireNdArray::from_i64(vec![3], vec![2, 12, 16]).unwrap(),
            offsets: None,
            cu_num_generated_tokens: None,
            logprobs: None,
        };
        let encoded = encode_msgpack(&wire).unwrap();
        let decoded: MaybeWireSamplingMask = decode_msgpack(&encoded).unwrap();
        let decoded = decoded.resolve(&[], "new_sampling_mask").unwrap();

        assert_eq!(decoded.into_direct().unwrap().rows, vec![vec![2, 12, 16]]);
    }

    #[test]
    fn decodes_legacy_mask_without_logprobs() {
        let token_ids = WireNdArray::from_i64(vec![2], vec![2, 12]).unwrap();
        let encoded =
            encode_msgpack(&(token_ids, None::<WireNdArray>, None::<Vec<usize>>)).unwrap();
        let decoded: MaybeWireSamplingMask = decode_msgpack(&encoded).unwrap();
        let mask = decoded.resolve(&[], "new_sampling_mask").unwrap();
        assert_eq!(mask.rows, vec![vec![2, 12]]);
        assert_eq!(mask.logprobs, None);
    }

    #[test]
    fn decodes_csr_rows_and_rejects_bad_terminal_offset() {
        let wire = WireSamplingMask {
            token_ids: WireNdArray::from_i64(vec![4], vec![2, 12, 16, 18]).unwrap(),
            offsets: Some(WireNdArray::from_i64(vec![3], vec![0, 2, 4]).unwrap()),
            cu_num_generated_tokens: None,
            logprobs: None,
        };
        let decoded = MaybeWireSamplingMask::Wire(Box::new(wire))
            .resolve(&[], "new_sampling_mask")
            .unwrap();
        assert_eq!(
            decoded.into_direct().unwrap().rows,
            vec![vec![2, 12], vec![16, 18]]
        );

        let malformed = WireSamplingMask {
            token_ids: WireNdArray::from_i64(vec![2], vec![2, 12]).unwrap(),
            offsets: Some(WireNdArray::from_i64(vec![2], vec![0, 1]).unwrap()),
            cu_num_generated_tokens: None,
            logprobs: None,
        };
        let error = MaybeWireSamplingMask::Wire(Box::new(malformed))
            .resolve(&[], "new_sampling_mask")
            .unwrap_err();
        assert!(error.to_string().contains("final offset"));
    }

    #[test]
    fn decodes_sampling_mask_from_multipart_aux_frames() {
        let output = EngineCoreOutput {
            request_id: "req-multipart-mask".to_string(),
            new_token_ids: vec![16, 18],
            new_sampling_mask: Some(MaybeWireSamplingMask::Wire(Box::new(WireSamplingMask {
                token_ids: WireNdArray {
                    dtype: NumpyDtype::little(TensorDtype::I32),
                    shape: vec![4],
                    data: WireArrayData::AuxIndex(1),
                },
                offsets: Some(WireNdArray {
                    dtype: NumpyDtype::little(TensorDtype::I64),
                    shape: vec![3],
                    data: WireArrayData::AuxIndex(2),
                }),
                cu_num_generated_tokens: None,
                logprobs: Some(WireNdArray {
                    dtype: NumpyDtype::little(TensorDtype::F32),
                    shape: vec![4],
                    data: WireArrayData::AuxIndex(3),
                }),
            }))),
            ..Default::default()
        };
        let primary = encode_msgpack(&crate::protocol::output::EngineCoreOutputs::from(
            RequestBatchOutputs {
                outputs: vec![output],
                ..Default::default()
            },
        ))
        .unwrap();
        let token_ids =
            [2_i32, 12, 16, 18].into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>();
        let offsets = [0_i64, 2, 4].into_iter().flat_map(i64::to_le_bytes).collect::<Vec<_>>();
        let logprobs = [-0.1_f32, -0.2, -0.3, -0.4]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let decoded = decode_engine_core_outputs(&[
            Bytes::from(primary),
            Bytes::from(token_ids),
            Bytes::from(offsets),
            Bytes::from(logprobs),
        ])
        .unwrap();
        let mask = decoded.as_request_batch().unwrap().outputs[0]
            .new_sampling_mask
            .as_ref()
            .unwrap();

        assert_eq!(mask.rows, vec![vec![2, 12], vec![16, 18]]);
        assert_eq!(
            mask.logprobs,
            Some(vec![vec![-0.1, -0.2], vec![-0.3, -0.4]])
        );
    }
}
