// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::mem::size_of;

use half::{bf16, f16};
use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::MmKwargValue as ProtocolKwargValue;
use vllm_engine_core_client::protocol::tensor::{ShapeExt as _, WireArrayData, WireTensor};

use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Representation for multimodal kwarg values for transformation.
#[derive(Debug)]
pub(super) enum KwargValue {
    /// Float tensor with row-major flat data and shape.
    F32Tensor(WireTensor),
    /// Float16 tensor with row-major flat data and shape.
    F16Tensor(WireTensor),
    /// BFloat16 tensor with row-major flat data and shape.
    Bf16Tensor(WireTensor),
    /// Signed integer tensor with row-major flat data and shape.
    I64Tensor(WireTensor),
    /// Unsigned integer tensor with row-major flat data and shape.
    U32Tensor(WireTensor),
    /// Non-tensor kwarg value that is shared or copied as-is.
    Passthrough(ProtocolKwargValue),
}

/// Collect the primary encoder input and model-specific outputs into one
/// tensor map.
///
/// `primary_key` names the encoder-input tensor as the model's forward kwargs
/// expect it (e.g. `pixel_values` for images, `pixel_values_videos` for
/// videos).
pub(super) fn collect_tensors(
    preprocessed: PreprocessedEncoderInputs,
    primary_key: &str,
    float_dtype: ModelDtype,
) -> Result<HashMap<String, KwargValue>> {
    let PreprocessedEncoderInputs {
        encoder_input,
        model_specific,
        ..
    } = preprocessed;

    let primary_value = {
        let shape = encoder_input.shape().to_vec();
        let data = encoder_input.into_iter().collect();
        KwargValue::from_f32_tensor(data, shape, float_dtype)?
    };

    let mut tensors = HashMap::new();
    tensors.insert(primary_key.to_string(), primary_value);
    for (key, value) in model_specific {
        tensors.insert(key, KwargValue::from_model_specific(value, float_dtype)?);
    }
    Ok(tensors)
}

impl KwargValue {
    fn from_model_specific(value: ModelSpecificValue, float_dtype: ModelDtype) -> Result<Self> {
        use ProtocolKwargValue::*;

        Ok(match value {
            ModelSpecificValue::Tensor { data, shape } => {
                Self::from_f32_tensor(data, shape, float_dtype)?
            }
            ModelSpecificValue::IntTensor { data, shape } => {
                Self::I64Tensor(WireTensor::from_owned_i64(shape, data).map_err(Error::Multimodal)?)
            }
            ModelSpecificValue::UintTensor { data, shape } => {
                Self::U32Tensor(WireTensor::from_owned_u32(shape, data).map_err(Error::Multimodal)?)
            }
            ModelSpecificValue::Int(value) => Self::Passthrough(Int(value)),
            ModelSpecificValue::Float(value) => Self::Passthrough(Float(value)),
            ModelSpecificValue::IntVec(values) => {
                Self::Passthrough(List(values.into_iter().map(Int).collect()))
            }
            ModelSpecificValue::UintVec(values) => Self::Passthrough(List(
                values.into_iter().map(|value| Int(value as i64)).collect(),
            )),
            ModelSpecificValue::FloatVec(values) => Self::Passthrough(List(
                values.into_iter().map(|value| Float(value as f64)).collect(),
            )),
            ModelSpecificValue::TupleVec(values) => Self::Passthrough(List(
                values
                    .into_iter()
                    .map(|(height, width)| List(vec![Int(height as i64), Int(width as i64)]))
                    .collect(),
            )),
            ModelSpecificValue::Bool(value) => Self::Passthrough(Int(i64::from(value))),
        })
    }

    /// Convert a float tensor to the target float dtype if needed, keeping the
    /// same shape.
    fn from_f32_tensor(data: Vec<f32>, shape: Vec<usize>, float_dtype: ModelDtype) -> Result<Self> {
        match float_dtype {
            ModelDtype::Float16 => {
                WireTensor::from_owned_f16(shape, data.into_iter().map(f16::from_f32).collect())
                    .map(Self::F16Tensor)
                    .map_err(Error::Multimodal)
            }
            ModelDtype::BFloat16 => {
                WireTensor::from_owned_bf16(shape, data.into_iter().map(bf16::from_f32).collect())
                    .map(Self::Bf16Tensor)
                    .map_err(Error::Multimodal)
            }
            ModelDtype::Float32 => WireTensor::from_owned_f32(shape, data)
                .map(Self::F32Tensor)
                .map_err(Error::Multimodal),
        }
    }
}

impl TryFrom<&KwargValue> for ProtocolKwargValue {
    type Error = Error;

    fn try_from(value: &KwargValue) -> Result<Self> {
        let tensor = match value {
            KwargValue::F32Tensor(tensor)
            | KwargValue::F16Tensor(tensor)
            | KwargValue::Bf16Tensor(tensor)
            | KwargValue::I64Tensor(tensor)
            | KwargValue::U32Tensor(tensor) => tensor.clone(),
            KwargValue::Passthrough(value) => return Ok(value.clone()),
        };
        Ok(ProtocolKwargValue::Tensor(tensor))
    }
}

impl KwargValue {
    /// First-axis length for tensor values; `None` for passthrough kwargs.
    pub(super) fn first_dim(&self) -> Option<usize> {
        match self {
            Self::F32Tensor(tensor)
            | Self::F16Tensor(tensor)
            | Self::Bf16Tensor(tensor)
            | Self::I64Tensor(tensor)
            | Self::U32Tensor(tensor) => tensor.shape.first().copied(),
            Self::Passthrough(_) => None,
        }
    }

    /// Convert one media item from a batched tensor field to wire bytes.
    ///
    /// Batched fields use their first axis as media-item index and drop that
    /// axis in the per-feature value, matching vLLM's batched-field semantics.
    pub(super) fn batched_wire_value_at(&self, index: usize) -> Result<ProtocolKwargValue> {
        self.wire_value_range(index, index + 1, true)
    }

    /// Convert one media item's flat tensor range directly to wire bytes.
    ///
    /// Flat fields keep the first axis as the sliced length for this item.
    pub(super) fn flat_wire_value_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<ProtocolKwargValue> {
        self.wire_value_range(start, end, false)
    }

    fn wire_value_range(
        &self,
        start: usize,
        end: usize,
        drop_axis: bool,
    ) -> Result<ProtocolKwargValue> {
        let tensor = match self {
            Self::F32Tensor(tensor) => {
                slice_first_axis_range(tensor, size_of::<f32>(), start, end, drop_axis)
            }
            Self::F16Tensor(tensor) => {
                slice_first_axis_range(tensor, size_of::<f16>(), start, end, drop_axis)
            }
            Self::Bf16Tensor(tensor) => {
                slice_first_axis_range(tensor, size_of::<bf16>(), start, end, drop_axis)
            }
            Self::I64Tensor(tensor) => {
                slice_first_axis_range(tensor, size_of::<i64>(), start, end, drop_axis)
            }
            Self::U32Tensor(tensor) => {
                slice_first_axis_range(tensor, size_of::<u32>(), start, end, drop_axis)
            }
            Self::Passthrough(value) => return Ok(value.clone()),
        };
        tensor.map(ProtocolKwargValue::Tensor)
    }
}

/// Compute the first-axis range for one media item in a flat tensor.
///
/// `sizes_key` names a companion tensor whose entries are cumulative slice
/// sizes per media item.
pub(super) fn flat_range_for_index(
    sizes: &KwargValue,
    sizes_key: &str,
    index: usize,
) -> Result<(usize, usize)> {
    let sizes = tensor_as_usize_vec(sizes)?;
    let size = *sizes.get(index).ok_or_else(|| {
        multimodal!("flat tensor sizes key `{sizes_key}` has no entry for media item {index}")
    })?;
    let start = sizes[..index].iter().sum::<usize>();
    Ok((start, start + size))
}

/// Read a tensor value as per-image sizes for flat slicing.
fn tensor_as_usize_vec(tensor: &KwargValue) -> Result<Vec<usize>> {
    match tensor {
        KwargValue::I64Tensor(tensor) => raw_tensor_bytes(tensor, size_of::<i64>())?
            .chunks_exact(size_of::<i64>())
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().expect("exact int64 chunk")))
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| multimodal!("negative flat tensor size `{value}`"))
            })
            .collect(),
        KwargValue::U32Tensor(tensor) => Ok(raw_tensor_bytes(tensor, size_of::<u32>())?
            .chunks_exact(size_of::<u32>())
            .map(|bytes| u32::from_ne_bytes(bytes.try_into().expect("exact uint32 chunk")) as usize)
            .collect()),
        _ => Err(multimodal!("flat tensor sizes must be int64 or uint32")),
    }
}

/// Slice a flat row-major tensor along its first axis.
fn slice_first_axis_range(
    tensor: &WireTensor,
    element_size: usize,
    start: usize,
    end: usize,
    drop_axis: bool,
) -> Result<WireTensor> {
    let shape = tensor.shape.as_slice();
    raw_tensor_bytes(tensor, element_size)?;
    let first_dim = *shape.first().ok_or_else(|| multimodal!("tensor has no first dimension"))?;
    if start > end || end > first_dim {
        bail_multimodal!("invalid tensor slice {start}..{end} for first dimension {first_dim}");
    }
    let stride = shape[1..]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .and_then(|stride| stride.checked_mul(element_size))
        .ok_or_else(|| multimodal!("tensor shape {shape:?} byte stride overflowed usize"))?;
    let data_start = start
        .checked_mul(stride)
        .ok_or_else(|| multimodal!("tensor slice start byte offset overflowed usize"))?;
    let data_end = end
        .checked_mul(stride)
        .ok_or_else(|| multimodal!("tensor slice end byte offset overflowed usize"))?;
    let out_shape = if drop_axis {
        shape[1..].to_vec()
    } else {
        let mut shape = shape.to_vec();
        shape[0] = end - start;
        shape
    };
    let WireArrayData::RawView(data) = &tensor.data else {
        return Err(multimodal!("cannot slice an aux tensor buffer"));
    };
    Ok(WireTensor::from_raw_bytes(
        tensor.dtype.clone(),
        out_shape,
        data.slice(data_start..data_end),
    ))
}

fn raw_tensor_bytes(tensor: &WireTensor, element_size: usize) -> Result<&[u8]> {
    let WireArrayData::RawView(data) = &tensor.data else {
        return Err(multimodal!("expected an inline tensor buffer"));
    };
    let expected_bytes = tensor
        .shape
        .checked_numel()
        .and_then(|numel| numel.checked_mul(element_size))
        .ok_or_else(|| {
            multimodal!(
                "tensor shape {:?} byte length overflowed usize",
                tensor.shape
            )
        })?;
    if expected_bytes != data.len() {
        bail_multimodal!(
            "tensor shape {:?} expects {expected_bytes} bytes, got {}",
            tensor.shape,
            data.len()
        );
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_wire_value_at_drops_first_axis() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let expected_ptr = data.as_ptr().cast::<u8>().wrapping_add(2 * size_of::<f32>());
        let value = KwargValue::F32Tensor(WireTensor::from_owned_f32(vec![2, 2], data).unwrap());

        let ProtocolKwargValue::Tensor(tensor) = value.batched_wire_value_at(1).unwrap() else {
            panic!("expected tensor");
        };

        assert_eq!(tensor.shape, vec![2]);
        let raw_view = tensor.data.into_raw_view().unwrap();
        assert_eq!(raw_view.as_ptr(), expected_ptr);
        assert_eq!(
            raw_view,
            [3.0_f32, 4.0].into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>()
        );
    }

    #[test]
    fn flat_wire_value_range_keeps_first_axis() {
        let value = KwargValue::U32Tensor(
            WireTensor::from_owned_u32(vec![5, 2], (0..10_u32).collect()).unwrap(),
        );

        let ProtocolKwargValue::Tensor(tensor) = value.flat_wire_value_range(1, 3).unwrap() else {
            panic!("expected tensor");
        };

        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(
            tensor.data.into_raw_view().unwrap(),
            [2_u32, 3, 4, 5].into_iter().flat_map(u32::to_ne_bytes).collect::<Vec<_>>()
        );
    }

    #[test]
    fn flat_range_for_index_uses_size_tensor() {
        let sizes =
            KwargValue::I64Tensor(WireTensor::from_owned_i64(vec![3], vec![2_i64, 3, 4]).unwrap());

        assert_eq!(
            flat_range_for_index(&sizes, "image_grid_thw", 1).unwrap(),
            (2, 5)
        );
    }

    #[test]
    fn slice_first_axis_range_errors_on_shape_data_mismatch() {
        let tensor = WireTensor::from_raw("float32", vec![2, 2], vec![0; 3 * size_of::<f32>()]);
        let error = slice_first_axis_range(&tensor, size_of::<f32>(), 0, 1, true).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expects 16 bytes"))
        );
    }

    #[test]
    fn bfloat16_tensor_wire_uses_bfloat16_dtype() {
        let value =
            KwargValue::from_f32_tensor(vec![1.0, -1.0], vec![2], ModelDtype::BFloat16).unwrap();

        let ProtocolKwargValue::Tensor(tensor) = ProtocolKwargValue::try_from(&value).unwrap()
        else {
            panic!("expected tensor");
        };

        assert_eq!(tensor.dtype, "bfloat16");
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(tensor.data.into_raw_view().unwrap().len(), 4);
    }

    #[test]
    fn float16_tensor_wire_uses_float16_dtype() {
        let value =
            KwargValue::from_f32_tensor(vec![1.0, -1.0], vec![2], ModelDtype::Float16).unwrap();

        let ProtocolKwargValue::Tensor(tensor) = ProtocolKwargValue::try_from(&value).unwrap()
        else {
            panic!("expected tensor");
        };

        assert_eq!(tensor.dtype, "float16");
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(tensor.data.into_raw_view().unwrap().len(), 4);
    }
}
