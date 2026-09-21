// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::Result;
use crate::protocol::logprobs::array::decode_array2_f32;
use crate::protocol::tensor::WireNdArray;

/// Log probabilities of `SamplingParams.prompt_logprob_token_ids`, row-major
/// with shape `[rows, cols]`: row `i` scores the candidates as predictions of
/// prompt token `prompt_logprob_start + i + 1`, column `j` is candidate `j`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptTokenIdLogprobs {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

/// Output field wrapper that is initially deserialized from the Python
/// `torch.Tensor` wire shape, then resolved into [`PromptTokenIdLogprobs`].
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum MaybeWirePromptTokenIdLogprobs {
    Wire(Box<WireNdArray>),
    Direct(PromptTokenIdLogprobs),
}

impl<'de> Deserialize<'de> for MaybeWirePromptTokenIdLogprobs {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireNdArray::deserialize(deserializer).map(|value| Self::Wire(Box::new(value)))
    }
}

impl Serialize for MaybeWirePromptTokenIdLogprobs {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // For testing purposes only. We don't actually serialize it into aux frames.
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(value) => {
                WireNdArray::from_f32(vec![value.rows, value.cols], value.data.clone())
                    .map_err(serde::ser::Error::custom)?
                    .serialize(serializer)
            }
        }
    }
}

impl MaybeWirePromptTokenIdLogprobs {
    pub(super) fn resolve(self, frames: &[Bytes], field_prefix: &str) -> Result<Self> {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => {
                let array = decode_array2_f32(*value, field_prefix, frames)?;
                Ok(Self::Direct(PromptTokenIdLogprobs {
                    rows: array.rows,
                    cols: array.cols,
                    data: array.data,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::dtype::{NumpyDtype, TensorDtype};
    use crate::protocol::encode_msgpack;
    use crate::protocol::output::{
        EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs, decode_engine_core_outputs,
    };
    use crate::protocol::tensor::WireArrayData;

    #[test]
    fn decodes_scores_from_multipart_aux_frame() {
        let output = EngineCoreOutput {
            request_id: "req-scores".to_string(),
            new_token_ids: vec![7],
            prompt_token_id_logprobs: Some(MaybeWirePromptTokenIdLogprobs::Wire(Box::new(
                WireNdArray {
                    dtype: NumpyDtype::little(TensorDtype::F32),
                    shape: vec![2, 3],
                    data: WireArrayData::AuxIndex(1),
                },
            ))),
            ..Default::default()
        };
        let primary = encode_msgpack(&EngineCoreOutputs::from(RequestBatchOutputs {
            outputs: vec![output],
            ..Default::default()
        }))
        .unwrap();
        let scores = [-0.5_f32, -1.5, -2.5, -3.5, -4.5, -5.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();

        let decoded =
            decode_engine_core_outputs(&[Bytes::from(primary), Bytes::from(scores)]).unwrap();
        let scores = decoded.as_request_batch().unwrap().outputs[0]
            .prompt_token_id_logprobs
            .as_ref()
            .unwrap()
            .as_direct()
            .unwrap();

        assert_eq!(
            scores,
            &PromptTokenIdLogprobs {
                rows: 2,
                cols: 3,
                data: vec![-0.5, -1.5, -2.5, -3.5, -4.5, -5.5],
            }
        );
    }
}
