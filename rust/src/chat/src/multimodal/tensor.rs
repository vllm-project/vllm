use std::collections::HashMap;

use llm_multimodal::{ModelSpecificValue, PreprocessedImages};
use vllm_engine_core_client::protocol::multimodal::MmKwargValue as ProtocolKwargValue;
use vllm_engine_core_client::protocol::tensor_wire::WireTensor;

use crate::error::{Error, Result, bail_multimodal, multimodal};

/// Representation for multimodal kwarg values for transformation.
#[derive(Clone)]
pub(super) enum KwargValue {
    /// Float tensor with row-major flat data and shape.
    F32Tensor { data: Vec<f32>, shape: Vec<usize> },
    /// Signed integer tensor with row-major flat data and shape.
    I64Tensor { data: Vec<i64>, shape: Vec<usize> },
    /// Unsigned integer tensor with row-major flat data and shape.
    U32Tensor { data: Vec<u32>, shape: Vec<usize> },
    /// Non-tensor kwarg value that is shared or copied as-is.
    Passthrough(ProtocolKwargValue),
}

/// Collect `pixel_values` and model-specific outputs into one tensor map.
pub(super) fn collect_tensors(preprocessed: PreprocessedImages) -> HashMap<String, KwargValue> {
    let PreprocessedImages {
        pixel_values,
        model_specific,
        ..
    } = preprocessed;

    let mut tensors = HashMap::from([(
        "pixel_values".to_string(),
        KwargValue::F32Tensor {
            shape: pixel_values.shape().to_vec(),
            data: pixel_values.into_iter().collect(),
        },
    )]);
    for (key, value) in model_specific {
        tensors.insert(key, KwargValue::from(value));
    }
    tensors
}

impl From<ModelSpecificValue> for KwargValue {
    fn from(value: ModelSpecificValue) -> Self {
        use ProtocolKwargValue::*;

        match value {
            ModelSpecificValue::Tensor { data, shape } => Self::F32Tensor { data, shape },
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
    /// Extract one image from a batched tensor field.
    ///
    /// Batched fields use their first axis as image index and drop that axis in
    /// the per-feature value, matching vLLM's batched-field semantics.
    pub(super) fn batched_value_at(&self, index: usize) -> Result<Self> {
        match self {
            Self::F32Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, index, index + 1, true)?;
                Ok(Self::F32Tensor { data, shape })
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

    /// Extract one image's variable-length range from a flat tensor field.
    ///
    /// Flat fields keep the first axis as the sliced length for this image.
    pub(super) fn flat_value_range(&self, start: usize, end: usize) -> Result<Self> {
        match self {
            Self::F32Tensor { data, shape } => {
                let (shape, data) = slice_first_axis_range(shape, data, start, end, false)?;
                Ok(Self::F32Tensor { data, shape })
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

/// Compute the first-axis range for one image in a flat tensor.
///
/// `sizes_key` names a companion tensor whose entries are cumulative slice
/// sizes per image.
pub(super) fn flat_range_for_index(
    sizes: &KwargValue,
    sizes_key: &str,
    index: usize,
) -> Result<(usize, usize)> {
    let sizes = tensor_as_usize_vec(sizes)?;
    let size = *sizes.get(index).ok_or_else(|| {
        multimodal!("flat tensor sizes key `{sizes_key}` has no entry for image {index}")
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
    let expected_len = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| multimodal!("tensor shape {shape:?} has too many elements"))
    })?;
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
}
