use std::collections::HashMap;

use half::{bf16, f16};
use llm_multimodal::{ModelSpecificValue, PreprocessedEncoderInputs};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::multimodal::MmKwargValue as ProtocolKwargValue;
use vllm_engine_core_client::protocol::tensor::{ShapeExt as _, WireTensor};

use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Representation for multimodal kwarg values for transformation.
#[derive(Debug, Clone)]
pub(super) enum KwargValue {
    /// Float tensor with row-major flat data and shape.
    F32Tensor { data: Vec<f32>, shape: Vec<usize> },
    /// Float16 tensor with row-major flat data and shape.
    F16Tensor { data: Vec<f16>, shape: Vec<usize> },
    /// BFloat16 tensor with row-major flat data and shape.
    Bf16Tensor { data: Vec<bf16>, shape: Vec<usize> },
    /// Signed integer tensor with row-major flat data and shape.
    I64Tensor { data: Vec<i64>, shape: Vec<usize> },
    /// Unsigned integer tensor with row-major flat data and shape.
    U32Tensor { data: Vec<u32>, shape: Vec<usize> },
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
            ModelSpecificValue::IntTensor { data, shape } => Self::I64Tensor { data, shape },
            ModelSpecificValue::UintTensor { data, shape } => Self::U32Tensor { data, shape },
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
            ModelDtype::Float16 => Ok(Self::F16Tensor {
                data: data.into_iter().map(f16::from_f32).collect(),
                shape,
            }),
            ModelDtype::BFloat16 => Ok(Self::Bf16Tensor {
                data: data.into_iter().map(bf16::from_f32).collect(),
                shape,
            }),
            ModelDtype::Float32 => Ok(Self::F32Tensor { data, shape }),
        }
    }
}

impl TryFrom<KwargValue> for ProtocolKwargValue {
    type Error = Error;

    fn try_from(value: KwargValue) -> Result<Self> {
        match value {
            KwargValue::F32Tensor { data, shape } => Ok(Self::Tensor(
                WireTensor::from_f32(shape, data).map_err(Error::Multimodal)?,
            )),
            KwargValue::F16Tensor { data, shape } => Ok(Self::Tensor(
                WireTensor::from_f16(shape, data).map_err(Error::Multimodal)?,
            )),
            KwargValue::Bf16Tensor { data, shape } => Ok(Self::Tensor(
                WireTensor::from_bf16(shape, data).map_err(Error::Multimodal)?,
            )),
            KwargValue::I64Tensor { data, shape } => Ok(Self::Tensor(
                WireTensor::from_i64(shape, data).map_err(Error::Multimodal)?,
            )),
            KwargValue::U32Tensor { data, shape } => Ok(Self::Tensor(
                WireTensor::from_u32(shape, data).map_err(Error::Multimodal)?,
            )),
            KwargValue::Passthrough(value) => Ok(value),
        }
    }
}

impl KwargValue {
    /// First-axis length for tensor values; `None` for passthrough kwargs.
    pub(super) fn first_dim(&self) -> Option<usize> {
        match self {
            Self::F32Tensor { shape, .. }
            | Self::F16Tensor { shape, .. }
            | Self::Bf16Tensor { shape, .. }
            | Self::I64Tensor { shape, .. }
            | Self::U32Tensor { shape, .. } => shape.first().copied(),
            Self::Passthrough(_) => None,
        }
    }

    /// Extract one media item from a batched tensor field.
    ///
    /// Batched fields use their first axis as media-item index and drop that
    /// axis in the per-feature value, matching vLLM's batched-field semantics.
    pub(super) fn batched_value_at(&self, index: usize) -> Result<Self> {
        match self {
            Self::F32Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
                Ok(Self::F32Tensor { data, shape })
            }
            Self::F16Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
                Ok(Self::F16Tensor { data, shape })
            }
            Self::Bf16Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
                Ok(Self::Bf16Tensor { data, shape })
            }
            Self::I64Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
                Ok(Self::I64Tensor { data, shape })
            }
            Self::U32Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
                Ok(Self::U32Tensor { data, shape })
            }
            Self::Passthrough(value) => Ok(Self::Passthrough(value.clone())),
        }
    }

    /// Extract one media item's variable-length range from a flat tensor field.
    ///
    /// Flat fields keep the first axis as the sliced length for this item.
    pub(super) fn flat_value_range(&self, start: usize, end: usize) -> Result<Self> {
        match self {
            Self::F32Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
                Ok(Self::F32Tensor { data, shape })
            }
            Self::F16Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
                Ok(Self::F16Tensor { data, shape })
            }
            Self::Bf16Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
                Ok(Self::Bf16Tensor { data, shape })
            }
            Self::I64Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
                Ok(Self::I64Tensor { data, shape })
            }
            Self::U32Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
                Ok(Self::U32Tensor { data, shape })
            }
            Self::Passthrough(value) => Ok(Self::Passthrough(value.clone())),
        }
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
        KwargValue::I64Tensor { data, .. } => data
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .map_err(|_| multimodal!("negative flat tensor size `{value}`"))
            })
            .collect(),
        KwargValue::U32Tensor { data, .. } => {
            Ok(data.iter().map(|value| *value as usize).collect())
        }
        _ => Err(multimodal!("flat tensor sizes must be int64 or uint32")),
    }
}

/// Slice a flat row-major tensor along its first axis.
fn slice_first_axis_range<T: Clone>(
    shape: &[usize],
    data: &[T],
    start: usize,
    end: usize,
    drop_axis: bool,
) -> Result<(Vec<usize>, Vec<T>)> {
    let first_dim = *shape.first().ok_or_else(|| multimodal!("tensor has no first dimension"))?;
    if start > end || end > first_dim {
        bail_multimodal!("invalid tensor slice {start}..{end} for first dimension {first_dim}");
    }
    let expected_len = shape
        .checked_numel()
        .ok_or_else(|| multimodal!("tensor shape {shape:?} has too many elements"))?;
    if expected_len != data.len() {
        bail_multimodal!(
            "tensor shape {shape:?} expects {expected_len} elements, got {}",
            data.len()
        );
    }
    let stride = shape[1..].iter().product::<usize>();
    let data_start = start * stride;
    let data_end = end * stride;
    let out_shape = if drop_axis {
        shape[1..].to_vec()
    } else {
        let mut shape = shape.to_vec();
        shape[0] = end - start;
        shape
    };
    Ok((out_shape, data[data_start..data_end].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_value_at_drops_first_axis() {
        let value = KwargValue::F32Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![2, 2],
        };

        let value = value.batched_value_at(1).unwrap();

        assert!(matches!(
            value,
            KwargValue::F32Tensor { data, shape }
                if shape == vec![2] && data == vec![3.0, 4.0]
        ));
    }

    #[test]
    fn flat_value_range_keeps_first_axis() {
        let value = KwargValue::U32Tensor {
            data: (0..10).collect(),
            shape: vec![5, 2],
        };

        let value = value.flat_value_range(1, 3).unwrap();

        assert!(matches!(
            value,
            KwargValue::U32Tensor { data, shape }
                if shape == vec![2, 2] && data == vec![2, 3, 4, 5]
        ));
    }

    #[test]
    fn flat_range_for_index_uses_size_tensor() {
        let sizes = KwargValue::I64Tensor {
            data: vec![2, 3, 4],
            shape: vec![3],
        };

        assert_eq!(
            flat_range_for_index(&sizes, "image_grid_thw", 1).unwrap(),
            (2, 5)
        );
    }

    #[test]
    fn slice_first_axis_range_errors_on_shape_data_mismatch() {
        let error = slice_first_axis_range(&[2, 2], &[1.0_f32, 2.0, 3.0], 0, 1, true).unwrap_err();

        assert!(
            matches!(error, Error::Multimodal(message) if message.contains("expects 4 elements"))
        );
    }

    #[test]
    fn bfloat16_tensor_wire_uses_bfloat16_dtype() {
        let value =
            KwargValue::from_f32_tensor(vec![1.0, -1.0], vec![2], ModelDtype::BFloat16).unwrap();

        let ProtocolKwargValue::Tensor(tensor) = ProtocolKwargValue::try_from(value).unwrap()
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

        let ProtocolKwargValue::Tensor(tensor) = ProtocolKwargValue::try_from(value).unwrap()
        else {
            panic!("expected tensor");
        };

        assert_eq!(tensor.dtype, "float16");
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(tensor.data.into_raw_view().unwrap().len(), 4);
    }
}
