// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::mem::size_of;

use half::{bf16, f16};
use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use ndarray::ArrayD;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::MmKwargValue as ProtocolKwargValue;
use vllm_engine_core_client::protocol::tensor::{ShapeExt as _, WireArrayData, WireTensor};

use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Element type retained alongside an encoded tensor during multimodal lowering.
#[derive(Debug, Clone, Copy)]
pub(super) enum TensorKind {
    /// 32-bit floating point.
    F32,
    /// IEEE 16-bit floating point.
    F16,
    /// Brain floating point.
    Bf16,
    /// Signed 64-bit integer.
    I64,
    /// Unsigned 32-bit integer.
    U32,
}

impl TensorKind {
    const fn element_size(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 => size_of::<f16>(),
            Self::Bf16 => size_of::<bf16>(),
            Self::I64 => size_of::<i64>(),
            Self::U32 => size_of::<u32>(),
        }
    }

    const fn wire_dtype(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::I64 => "int64",
            Self::U32 => "uint32",
        }
    }
}

/// Representation for multimodal kwarg values for transformation.
#[derive(Debug)]
pub(super) enum KwargValue {
    /// Tensor with row-major flat data and shape.
    Tensor { kind: TensorKind, wire: WireTensor },
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

    let primary_value = KwargValue::from_f32_array(encoder_input, float_dtype)?;

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
                Self::from_f32_parts(data, shape, float_dtype)?
            }
            ModelSpecificValue::IntTensor { data, shape } => {
                let wire = WireTensor::from_i64(shape, data).map_err(Error::Multimodal)?;
                Self::tensor(TensorKind::I64, wire)
            }
            ModelSpecificValue::UintTensor { data, shape } => {
                let wire = WireTensor::from_u32(shape, data).map_err(Error::Multimodal)?;
                Self::tensor(TensorKind::U32, wire)
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

    fn from_f32_array(array: ArrayD<f32>, float_dtype: ModelDtype) -> Result<Self> {
        let shape = array.shape().to_vec();
        // `into_iter()` on dynamically-dimensioned arrays walks a slow
        // per-element path; take the raw buffer instead when the layout
        // allows it.
        let data = if array.is_standard_layout() {
            let len = array.len();
            let (data, offset) = array.into_raw_vec_and_offset();
            let start = offset.unwrap_or(0);
            if start == 0 && data.len() == len {
                data
            } else {
                // Buffer with unused head/tail: copy the used range, which
                // standard strides place contiguously in logical order.
                data[start..start + len].to_vec()
            }
        } else {
            array.into_iter().collect()
        };
        Self::from_f32_parts(data, shape, float_dtype)
    }

    /// Convert a float tensor to the target float dtype if needed, keeping the
    /// same shape.
    fn from_f32_parts(data: Vec<f32>, shape: Vec<usize>, float_dtype: ModelDtype) -> Result<Self> {
        let (kind, wire) = match float_dtype {
            ModelDtype::Float16 => (
                TensorKind::F16,
                WireTensor::from_f16(shape, data.into_iter().map(f16::from_f32).collect()),
            ),
            ModelDtype::BFloat16 => (
                TensorKind::Bf16,
                WireTensor::from_bf16(shape, data.into_iter().map(bf16::from_f32).collect()),
            ),
            ModelDtype::Float32 => (TensorKind::F32, WireTensor::from_f32(shape, data)),
        };
        wire.map(|wire| Self::tensor(kind, wire)).map_err(Error::Multimodal)
    }

    fn tensor(kind: TensorKind, wire: WireTensor) -> Self {
        debug_assert_eq!(wire.dtype, kind.wire_dtype());
        Self::Tensor { kind, wire }
    }
}

impl TryFrom<&KwargValue> for ProtocolKwargValue {
    type Error = Error;

    fn try_from(value: &KwargValue) -> Result<Self> {
        let wire = match value {
            KwargValue::Tensor { wire, .. } => wire.clone(),
            KwargValue::Passthrough(value) => return Ok(value.clone()),
        };
        Ok(ProtocolKwargValue::Tensor(wire))
    }
}

impl KwargValue {
    /// First-axis length for tensor values; `None` for passthrough kwargs.
    pub(super) fn first_dim(&self) -> Option<usize> {
        match self {
            Self::Tensor { wire, .. } => wire.shape.first().copied(),
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
        let wire = match self {
            Self::Tensor { kind, wire } => {
                slice_first_axis_range(wire, kind.element_size(), start, end, drop_axis)
            }
            Self::Passthrough(value) => return Ok(value.clone()),
        };
        wire.map(ProtocolKwargValue::Tensor)
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
        KwargValue::Tensor {
            kind: TensorKind::I64,
            wire,
        } => raw_tensor_bytes(wire, size_of::<i64>())?
            .chunks_exact(size_of::<i64>())
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().expect("exact int64 chunk")))
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| multimodal!("negative flat tensor size `{value}`"))
            })
            .collect(),
        KwargValue::Tensor {
            kind: TensorKind::U32,
            wire,
        } => Ok(raw_tensor_bytes(wire, size_of::<u32>())?
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
    use ndarray::{Array2, s};

    #[test]
    fn collect_tensors_lowering_matches_logical_order_across_layouts() {
        let standard =
            Array2::from_shape_vec((2, 3), (1..=6).map(|v| v as f32).collect::<Vec<f32>>())
                .unwrap();
        let standard_with_offset =
            Array2::from_shape_vec((3, 3), (1..=9).map(|v| v as f32).collect::<Vec<f32>>())
                .unwrap()
                .slice_move(s![1.., ..]);
        let strided =
            Array2::from_shape_vec((2, 3), (1..=6).map(|v| v as f32).collect::<Vec<f32>>())
                .unwrap()
                .reversed_axes();

        assert!(standard.is_standard_layout());
        assert!(standard_with_offset.is_standard_layout());
        assert!(!strided.is_standard_layout());

        for (name, array) in [
            ("standard", standard),
            ("standard_with_offset", standard_with_offset),
            ("strided", strided),
        ] {
            let expected: Vec<u8> = array.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let shape = array.shape().to_vec();
            let preprocessed = PreprocessedEncoderInputs::new(array, vec![6], vec![(3, 2)]);
            let tensors =
                collect_tensors(preprocessed, "pixel_values", ModelDtype::Float32).unwrap();

            let ProtocolKwargValue::Tensor(tensor) =
                ProtocolKwargValue::try_from(&tensors["pixel_values"]).unwrap()
            else {
                panic!("expected tensor for {name} layout");
            };

            assert_eq!(tensor.shape, shape, "{name} layout shape");
            assert_eq!(tensor.dtype, "float32", "{name} layout dtype");
            assert_eq!(
                tensor.data.into_raw_view().unwrap(),
                expected,
                "{name} layout data"
            );
        }
    }

    #[test]
    fn batched_wire_value_at_drops_first_axis() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let expected_ptr = data.as_ptr().cast::<u8>().wrapping_add(2 * size_of::<f32>());
        let value = KwargValue::tensor(
            TensorKind::F32,
            WireTensor::from_f32(vec![2, 2], data).unwrap(),
        );

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
        let value = KwargValue::tensor(
            TensorKind::U32,
            WireTensor::from_u32(vec![5, 2], (0..10_u32).collect()).unwrap(),
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
        let sizes = KwargValue::tensor(
            TensorKind::I64,
            WireTensor::from_i64(vec![3], vec![2_i64, 3, 4]).unwrap(),
        );

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
            KwargValue::from_f32_parts(vec![1.0, -1.0], vec![2], ModelDtype::BFloat16).unwrap();

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
            KwargValue::from_f32_parts(vec![1.0, -1.0], vec![2], ModelDtype::Float16).unwrap();

        let ProtocolKwargValue::Tensor(tensor) = ProtocolKwargValue::try_from(&value).unwrap()
        else {
            panic!("expected tensor");
        };

        assert_eq!(tensor.dtype, "float16");
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(tensor.data.into_raw_view().unwrap().len(), 4);
    }
}
