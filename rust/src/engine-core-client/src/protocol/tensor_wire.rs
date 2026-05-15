use bytemuck::allocation::pod_collect_to_vec;
use enum_as_inner::EnumAsInner;
use rmpv::Value;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

/// Tensors and ndarrays are encoded with this extension type in Python.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L41-L43>
const CUSTOM_TYPE_RAW_VIEW: i8 = 3;

/// Python ndarray/tensor wire tuple encoded as `(dtype, shape, data)`.
///
/// This matches the custom msgpack representation built by Python
/// `serial_utils.encode_ndarray` / `encode_tensor`.
///
/// Original Python wire encoders:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L237-L273>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L389-L425>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireNdArray {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: WireArrayData,
}

impl WireNdArray {
    /// Build a float32 tensor/ndarray backed by native-endian raw-view bytes.
    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self {
            dtype: "float32".to_string(),
            shape,
            data: WireArrayData::RawView(pod_collect_to_vec::<f32, u8>(&data)),
        })
    }

    /// Build an int64 tensor/ndarray backed by native-endian raw-view bytes.
    pub fn from_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self {
            dtype: "int64".to_string(),
            shape,
            data: WireArrayData::RawView(pod_collect_to_vec::<i64, u8>(&data)),
        })
    }

    /// Build a uint32 tensor/ndarray backed by native-endian raw-view bytes.
    pub fn from_u32(shape: Vec<usize>, data: Vec<u32>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self {
            dtype: "uint32".to_string(),
            shape,
            data: WireArrayData::RawView(pod_collect_to_vec::<u32, u8>(&data)),
        })
    }

    /// Build a bool tensor/ndarray backed by raw-view bytes.
    ///
    /// This matches `torch.bool` storage: one byte per element, not a packed
    /// bitmap. Values are canonicalized as `false -> 0` and `true -> 1`.
    pub fn from_bool(shape: Vec<usize>, data: Vec<bool>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self {
            dtype: "bool".to_string(),
            shape,
            data: WireArrayData::RawView(data.iter().map(|value| u8::from(*value)).collect()),
        })
    }

    /// Build a tensor/ndarray from already-encoded raw-view bytes.
    ///
    /// Use this as an escape hatch when the caller already owns bytes that
    /// match the requested `dtype` and `shape`.
    pub fn from_raw(dtype: impl Into<String>, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            dtype: dtype.into(),
            shape,
            data: WireArrayData::RawView(data),
        }
    }
}

/// Validate that the number of elements implied by the shape matches the length
/// of the data.
fn validate_element_count(shape: &[usize], len: usize) -> Result<(), String> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| format!("tensor shape product overflows usize: {shape:?}"))?;
    if expected == len {
        Ok(())
    } else {
        Err(format!(
            "tensor data length {len} does not match shape {shape:?} product {expected}"
        ))
    }
}

/// Python tensor wire tuple encoded as `(dtype, shape, data)`.
///
/// This is the same wire shape as [`WireNdArray`]; multimodal request payloads
/// use it for `torch.Tensor` values.
pub type WireTensor = WireNdArray;

/// Python array/tensor payload reference inside [`WireNdArray`].
///
/// The data can be either an inline msgpack raw-view extension or an index into
/// the multipart aux-frame list carried alongside the primary msgpack frame.
///
/// Original Python wire encoders:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L237-L273>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L389-L425>
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum WireArrayData {
    /// The index of the aux frame where the raw bytes of this array/tensor are
    /// stored.
    AuxIndex(usize),
    /// The raw bytes of this array/tensor.
    RawView(Vec<u8>),
}

impl<'de> Deserialize<'de> for WireArrayData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Ext(tag, bytes) if tag == CUSTOM_TYPE_RAW_VIEW => Ok(Self::RawView(bytes)),
            Value::Ext(tag, _) => Err(serde::de::Error::custom(format!(
                "unsupported extension type code {tag}"
            ))),
            Value::Integer(index) => {
                index.as_u64().map(|index| Self::AuxIndex(index as usize)).ok_or_else(|| {
                    serde::de::Error::custom("aux frame index must be a non-negative integer")
                })
            }
            other => Err(serde::de::Error::custom(format!(
                "expected raw-view ext or aux frame index, got {other:?}"
            ))),
        }
    }
}

impl Serialize for WireArrayData {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // TODO: outbound request serialization currently only supports inline
        // raw-view bytes. Emitting aux frames needs transport-level plumbing;
        // serializing `AuxIndex` here only preserves an already-built reference.
        match self {
            Self::AuxIndex(index) => serializer.serialize_u64(*index as u64),
            Self::RawView(bytes) => {
                Value::Ext(CUSTOM_TYPE_RAW_VIEW, bytes.clone()).serialize(serializer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_build_raw_view_tensors() {
        let f32_tensor = WireNdArray::from_f32(vec![2], vec![1.0, 2.5]).unwrap();
        assert_eq!(f32_tensor.dtype, "float32");
        assert_eq!(f32_tensor.shape, vec![2]);
        assert_eq!(
            f32_tensor.data.into_raw_view().expect("raw view"),
            [1.0_f32, 2.5].into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>()
        );

        let i64_tensor = WireNdArray::from_i64(vec![1], vec![-7]).unwrap();
        assert_eq!(i64_tensor.dtype, "int64");
        assert_eq!(
            i64_tensor.data.into_raw_view().expect("raw view"),
            (-7_i64).to_ne_bytes()
        );

        let u32_tensor = WireNdArray::from_u32(vec![1], vec![42]).unwrap();
        assert_eq!(u32_tensor.dtype, "uint32");
        assert_eq!(
            u32_tensor.data.into_raw_view().expect("raw view"),
            42_u32.to_ne_bytes()
        );

        let bool_tensor = WireNdArray::from_bool(vec![2], vec![false, true]).unwrap();
        assert_eq!(bool_tensor.dtype, "bool");
        assert_eq!(
            bool_tensor.data.into_raw_view().expect("raw view"),
            vec![0, 1]
        );

        let raw_tensor = WireNdArray::from_raw("custom", vec![3], vec![1, 2, 3]);
        assert_eq!(raw_tensor.dtype, "custom");
        assert_eq!(raw_tensor.shape, vec![3]);
        assert_eq!(
            raw_tensor.data.into_raw_view().expect("raw view"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn constructors_validate_shape_product() {
        let err = WireNdArray::from_f32(vec![2, 2], vec![1.0, 2.0]).unwrap_err();
        assert!(err.contains("does not match shape"));
    }
}
