// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use bytemuck::{Pod, cast_slice};
use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use half::{bf16, f16};
use rmpv::Value;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::dtype::{Endianness, NumpyDtype, TensorDtype};

/// Tensors and ndarrays are encoded with this extension type in Python.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L41-L43>
const CUSTOM_TYPE_RAW_VIEW: i8 = 3;

#[derive(Serialize)]
#[serde(rename = "_ExtStruct")]
struct MsgpackExtRef<'a>((i8, ByteSlice<'a>));

struct ByteSlice<'a>(&'a [u8]);

struct PodVec<T: Pod>(Vec<T>);

impl<T: Pod> AsRef<[u8]> for PodVec<T> {
    fn as_ref(&self) -> &[u8] {
        cast_slice(&self.0)
    }
}

fn bytes_from_pod_vec<T>(data: Vec<T>) -> Bytes
where
    T: Pod + Send + 'static,
{
    Bytes::from_owner(PodVec(data))
}

impl Serialize for ByteSlice<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.0)
    }
}

#[easy_ext::ext(ShapeExt)]
impl [usize] {
    /// Returned the total number of elements implied by this shape, or `None`
    /// if the product of the dimensions overflows `usize`.
    pub fn checked_numel(&self) -> Option<usize> {
        self.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
    }
}

/// Python ndarray/tensor wire tuple encoded as `(dtype, shape, data)`.
///
/// This matches the custom msgpack representation built by Python
/// `serial_utils.encode_ndarray` / `encode_tensor`.
///
/// Original Python wire encoders:
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L237-L273>
/// <https://github.com/vllm-project/vllm/blob/5a0a8fc1ea7542394ff315138bd5677b7b53bca1/vllm/v1/serial_utils.py#L389-L425>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireArray<Dtype: Serialize + DeserializeOwned> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: WireArrayData,
}

impl<Dtype: From<TensorDtype> + Serialize + DeserializeOwned> WireArray<Dtype> {
    /// Build a float32 tensor/ndarray backed by native-endian raw-view bytes.
    ///
    /// Takes ownership of the backing buffer without copying its data.
    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            TensorDtype::F32,
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// Build a float16 tensor/ndarray backed by native-endian raw-view bytes.
    ///
    /// Takes ownership of the backing buffer without copying its data.
    pub fn from_f16(shape: Vec<usize>, data: Vec<f16>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            TensorDtype::F16,
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// Build a bfloat16 tensor/ndarray backed by native-endian raw-view bytes.
    ///
    /// Takes ownership of the backing buffer without copying its data.
    pub fn from_bf16(shape: Vec<usize>, data: Vec<bf16>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            TensorDtype::Bf16,
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// Build an int64 tensor/ndarray backed by native-endian raw-view bytes.
    ///
    /// Takes ownership of the backing buffer without copying its data.
    pub fn from_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            TensorDtype::I64,
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// Build an int32 tensor/ndarray backed by native-endian raw-view bytes.
    ///
    /// Takes ownership of the backing buffer without copying its data.
    pub fn from_i32(shape: Vec<usize>, data: Vec<i32>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            TensorDtype::I32,
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// Build a uint32 tensor/ndarray backed by native-endian raw-view bytes.
    ///
    /// Takes ownership of the backing buffer without copying its data.
    pub fn from_u32(shape: Vec<usize>, data: Vec<u32>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            TensorDtype::U32,
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// Build a bool tensor/ndarray backed by raw-view bytes.
    ///
    /// This matches `torch.bool` storage: one byte per element, not a packed
    /// bitmap. Values are canonicalized as `false -> 0` and `true -> 1`.
    pub fn from_bool(shape: Vec<usize>, data: Vec<bool>) -> Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self {
            dtype: TensorDtype::Bool.into(),
            shape,
            data: WireArrayData::RawView(Bytes::from(
                data.into_iter().map(u8::from).collect::<Vec<_>>(),
            )),
        })
    }

    /// Build a tensor/ndarray from already-encoded raw-view bytes.
    ///
    /// Use this as an escape hatch when the caller already owns bytes that
    /// match the requested `dtype` and `shape`.
    pub fn from_raw(dtype: impl Into<Dtype>, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self::from_raw_bytes(dtype, shape, Bytes::from(data))
    }

    /// Build a tensor/ndarray from an owned immutable raw-view buffer.
    pub fn from_raw_bytes(dtype: impl Into<Dtype>, shape: Vec<usize>, data: Bytes) -> Self {
        Self {
            dtype: dtype.into(),
            shape,
            data: WireArrayData::RawView(data),
        }
    }

    /// Move a sufficiently large inline buffer into the ordered auxiliary-frame list.
    pub(crate) fn extract_aux_frame(&mut self, aux_frames: &mut Vec<Bytes>, threshold: usize) {
        self.data.extract_aux_frame(aux_frames, threshold);
    }

    /// Resolve an auxiliary-frame reference into owned raw bytes.
    pub(crate) fn resolve_aux_frame(&mut self, frames: &[Bytes]) -> Result<(), String> {
        self.data.resolve_aux_frame(frames)
    }
}

/// Validate that the number of elements implied by the shape matches the length
/// of the data.
fn validate_element_count(shape: &[usize], len: usize) -> Result<(), String> {
    let expected = shape
        .checked_numel()
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
pub type WireTensor = WireArray<TensorDtype>;

/// Numeric array with a typed NumPy descriptor or native-endian Torch dtype.
pub type WireNdArray = WireArray<NumpyDtype>;

/// A row operation on a [`WireNdArray`] that the array does not support.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArrayRowsError {
    #[error("a 0-d array has no rows")]
    NoRowAxis,
    #[error("array row size overflows: {shape:?}")]
    RowSizeOverflow { shape: Vec<usize> },
    #[error("array data is an unresolved auxiliary frame")]
    UnresolvedAuxFrame,
    #[error(
        "cannot concatenate {dtype:?} {shape:?} rows onto {first_dtype:?} {first_shape:?} rows"
    )]
    Mismatch {
        dtype: NumpyDtype,
        shape: Vec<usize>,
        first_dtype: NumpyDtype,
        first_shape: Vec<usize>,
    },
}

impl WireNdArray {
    /// Bytes per row along axis 0, for an array with at least one axis.
    fn row_bytes(&self) -> Result<usize, ArrayRowsError> {
        let row_shape = self.shape.get(1..).ok_or(ArrayRowsError::NoRowAxis)?;
        row_shape
            .checked_numel()
            .and_then(|numel| numel.checked_mul(self.dtype.scalar.element_size()))
            .ok_or_else(|| ArrayRowsError::RowSizeOverflow {
                shape: self.shape.clone(),
            })
    }

    fn raw_bytes(&self) -> Result<&[u8], ArrayRowsError> {
        self.data
            .as_raw_view()
            .map(|data| data.as_ref())
            .ok_or(ArrayRowsError::UnresolvedAuxFrame)
    }

    /// Keep the first `num_rows` rows along axis 0, like NumPy's
    /// `array[:num_rows]`; arrays with no more rows are unchanged.
    pub fn truncate_rows(&mut self, num_rows: usize) -> Result<(), ArrayRowsError> {
        let row_bytes = self.row_bytes()?;
        if num_rows >= self.shape[0] {
            return Ok(());
        }
        let WireArrayData::RawView(data) = &mut self.data else {
            return Err(ArrayRowsError::UnresolvedAuxFrame);
        };
        data.truncate(num_rows * row_bytes);
        self.shape[0] = num_rows;
        Ok(())
    }

    /// Concatenate arrays along axis 0, like NumPy's `np.concatenate`; `None`
    /// for no arrays. The arrays must share their dtype and row shape.
    pub fn concat_rows(
        arrays: impl IntoIterator<Item = Self>,
    ) -> Result<Option<Self>, ArrayRowsError> {
        let mut arrays = arrays.into_iter();
        let Some(first) = arrays.next() else {
            return Ok(None);
        };
        first.row_bytes()?;
        first.raw_bytes()?;
        let mut rest = arrays.peekable();
        if rest.peek().is_none() {
            return Ok(Some(first));
        }
        let mut num_rows = first.shape[0];
        let mut data = first.raw_bytes()?.to_vec();
        for array in rest {
            if array.dtype != first.dtype || array.shape.get(1..) != first.shape.get(1..) {
                return Err(ArrayRowsError::Mismatch {
                    dtype: array.dtype,
                    shape: array.shape,
                    first_dtype: first.dtype,
                    first_shape: first.shape,
                });
            }
            num_rows += array.shape[0];
            data.extend_from_slice(array.raw_bytes()?);
        }
        let mut shape = first.shape;
        shape[0] = num_rows;
        Ok(Some(Self::from_raw(first.dtype, shape, data)))
    }
}

/// A rank-1 uint32 array, such as prompt token ids.
impl From<Vec<u32>> for WireNdArray {
    fn from(data: Vec<u32>) -> Self {
        Self::from_u32(vec![data.len()], data).expect("rank-1 shape matches the data")
    }
}

impl WireNdArray {
    /// The values of an inline 32-bit integer array.
    pub fn to_u32_vec(&self) -> Option<Vec<u32>> {
        if !matches!(self.dtype.scalar, TensorDtype::U32 | TensorDtype::I32) {
            return None;
        }
        let decode = match self.dtype.endianness {
            Endianness::Little => u32::from_le_bytes,
            Endianness::Big => u32::from_be_bytes,
            Endianness::Native => u32::from_ne_bytes,
        };
        let bytes = self.data.as_raw_view()?;
        Some(bytes.chunks_exact(4).map(|c| decode(c.try_into().unwrap())).collect())
    }
}

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
    RawView(Bytes),
}

impl WireTensor {
    /// Validate the shape and storage of an inline tensor.
    pub fn validate_inline(&self) -> Result<(), String> {
        let expected = self
            .shape
            .checked_numel()
            .and_then(|numel| numel.checked_mul(self.dtype.element_size()))
            .ok_or_else(|| format!("tensor shape or byte length overflows: {:?}", self.shape))?;
        match &self.data {
            WireArrayData::RawView(bytes) if bytes.len() == expected => Ok(()),
            WireArrayData::RawView(bytes) => Err(format!(
                "tensor byte length {} does not match expected {expected}",
                bytes.len()
            )),
            WireArrayData::AuxIndex(index) => {
                Err(format!("inline tensor references auxiliary frame {index}"))
            }
        }
    }
}

impl WireArrayData {
    /// Resolve an auxiliary-frame reference into owned raw bytes.
    pub(crate) fn resolve_aux_frame(&mut self, frames: &[Bytes]) -> Result<(), String> {
        let Self::AuxIndex(index) = self else {
            return Ok(());
        };
        let index = *index;
        let frame = frames.get(index).ok_or_else(|| {
            format!(
                "auxiliary frame index {index} is out of range for {} frames",
                frames.len()
            )
        })?;
        *self = Self::RawView(frame.clone());
        Ok(())
    }

    /// Replace a sufficiently large raw view with its one-based auxiliary-frame index.
    fn extract_aux_frame(&mut self, aux_frames: &mut Vec<Bytes>, threshold: usize) {
        let Self::RawView(bytes) = self else {
            return;
        };
        if bytes.len() < threshold {
            return;
        }

        let index = aux_frames.len() + 1;
        let bytes = std::mem::replace(self, Self::AuxIndex(index))
            .into_raw_view()
            .expect("raw view was matched above");
        aux_frames.push(bytes);
    }
}

impl<'de> Deserialize<'de> for WireArrayData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Ext(tag, bytes) if tag == CUSTOM_TYPE_RAW_VIEW => {
                Ok(Self::RawView(Bytes::from(bytes)))
            }
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
        match self {
            Self::AuxIndex(index) => serializer.serialize_u64(*index as u64),
            Self::RawView(bytes) => {
                MsgpackExtRef((CUSTOM_TYPE_RAW_VIEW, ByteSlice(bytes))).serialize(serializer)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `[rows, 2]` u8 array with one `[value, 0]` row per value.
    fn rows(values: &[u8]) -> WireNdArray {
        WireNdArray::from_raw(
            TensorDtype::U8,
            vec![values.len(), 2],
            values.iter().flat_map(|&value| [value, 0]).collect(),
        )
    }

    #[test]
    fn truncate_rows_keeps_leading_rows() {
        let mut array = rows(&[1, 2, 3]);
        array.truncate_rows(5).unwrap();
        assert_eq!(array, rows(&[1, 2, 3]));
        array.truncate_rows(1).unwrap();
        assert_eq!(array, rows(&[1]));
        array.truncate_rows(0).unwrap();
        assert_eq!(array, rows(&[]));

        let mut scalar = WireNdArray::from_raw(TensorDtype::U8, vec![], vec![7]);
        assert!(scalar.truncate_rows(0).is_err());
        let mut unresolved = rows(&[1]);
        unresolved.data = WireArrayData::AuxIndex(1);
        assert!(unresolved.truncate_rows(0).is_err());
    }

    #[test]
    fn concat_rows_joins_along_axis_0() {
        assert_eq!(WireNdArray::concat_rows([]).unwrap(), None);
        assert_eq!(
            WireNdArray::concat_rows([rows(&[1, 2]), rows(&[]), rows(&[3])]).unwrap(),
            Some(rows(&[1, 2, 3]))
        );
        let other_row_shape = WireNdArray::from_raw(TensorDtype::U8, vec![1, 1], vec![0]);
        assert!(WireNdArray::concat_rows([rows(&[1]), other_row_shape]).is_err());
        let other_dtype = WireNdArray::from_raw(TensorDtype::I8, vec![1, 2], vec![0, 0]);
        assert!(WireNdArray::concat_rows([rows(&[1]), other_dtype]).is_err());

        // A single array is validated like several.
        let scalar = WireNdArray::from_raw(TensorDtype::U8, vec![], vec![7]);
        assert!(WireNdArray::concat_rows([scalar]).is_err());
        let mut unresolved = rows(&[1]);
        unresolved.data = WireArrayData::AuxIndex(1);
        assert!(WireNdArray::concat_rows([unresolved]).is_err());
    }

    #[test]
    fn tensor_and_ndarray_preserve_their_dtype_wire_contracts() {
        let tensor = WireTensor::from_f32(vec![1], vec![1.0]).unwrap();
        let bytes = rmp_serde::to_vec_named(&tensor).unwrap();
        let wire: Value = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(wire.as_array().unwrap()[0], Value::from("float32"));

        let array: WireNdArray = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(array.dtype, NumpyDtype::native(TensorDtype::F32));
        let bytes = rmp_serde::to_vec_named(&array).unwrap();
        let wire: Value = rmp_serde::from_slice(&bytes).unwrap();
        // Native byte order is spelled as the plain name, which a tensor
        // reads too: Python sends tensor fields this way.
        assert_eq!(wire.as_array().unwrap()[0], Value::from("float32"));
        let tensor: WireTensor = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(tensor.dtype, TensorDtype::F32);

        // An explicit byte order is a NumPy descriptor only.
        let array = WireNdArray::from_raw_bytes(
            NumpyDtype::little(TensorDtype::F32),
            vec![1],
            Bytes::from_static(&[0; 4]),
        );
        let bytes = rmp_serde::to_vec_named(&array).unwrap();
        let wire: Value = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(wire.as_array().unwrap()[0], Value::from("<f4"));
        assert!(rmp_serde::from_slice::<WireTensor>(&bytes).is_err());
    }

    #[test]
    fn raw_view_serializes_as_msgpack_ext() {
        let bytes = vec![1, 2, 3, 4];
        let encoded =
            rmp_serde::to_vec_named(&WireArrayData::RawView(bytes.clone().into())).expect("encode");
        let expected = rmp_serde::to_vec_named(&Value::Ext(CUSTOM_TYPE_RAW_VIEW, bytes.clone()))
            .expect("encode expected");

        assert_eq!(encoded, expected);
        assert_eq!(
            rmpv::decode::read_value(&mut std::io::Cursor::new(encoded)).expect("decode"),
            Value::Ext(CUSTOM_TYPE_RAW_VIEW, bytes)
        );
    }

    #[test]
    fn constructors_build_raw_view_tensors() {
        let f32_data = vec![1.0, 2.5];
        let f32_data_ptr = f32_data.as_ptr().cast::<u8>();
        let f32_tensor = WireTensor::from_f32(vec![2], f32_data).unwrap();
        assert_eq!(f32_tensor.dtype, TensorDtype::F32);
        assert_eq!(f32_tensor.shape, vec![2]);
        let f32_raw_view = f32_tensor.data.into_raw_view().expect("raw view");
        assert_eq!(f32_raw_view.as_ptr(), f32_data_ptr);
        assert_eq!(
            f32_raw_view,
            [1.0_f32, 2.5].into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>()
        );

        let f16_tensor =
            WireTensor::from_f16(vec![2], vec![f16::from_f32(1.0), f16::from_f32(2.5)]).unwrap();
        assert_eq!(f16_tensor.dtype, TensorDtype::F16);
        assert_eq!(f16_tensor.shape, vec![2]);
        assert_eq!(f16_tensor.data.into_raw_view().expect("raw view").len(), 4);

        let bf16_tensor =
            WireTensor::from_bf16(vec![2], vec![bf16::from_f32(1.0), bf16::from_f32(2.5)]).unwrap();
        assert_eq!(bf16_tensor.dtype, TensorDtype::Bf16);
        assert_eq!(bf16_tensor.shape, vec![2]);
        assert_eq!(bf16_tensor.data.into_raw_view().expect("raw view").len(), 4);

        let i64_tensor = WireTensor::from_i64(vec![1], vec![-7]).unwrap();
        assert_eq!(i64_tensor.dtype, TensorDtype::I64);
        assert_eq!(
            i64_tensor.data.into_raw_view().expect("raw view").as_ref(),
            (-7_i64).to_ne_bytes().as_ref()
        );

        let u32_tensor = WireTensor::from_u32(vec![1], vec![42]).unwrap();
        assert_eq!(u32_tensor.dtype, TensorDtype::U32);
        assert_eq!(
            u32_tensor.data.into_raw_view().expect("raw view").as_ref(),
            42_u32.to_ne_bytes().as_ref()
        );

        let bool_tensor = WireTensor::from_bool(vec![2], vec![false, true]).unwrap();
        assert_eq!(bool_tensor.dtype, TensorDtype::Bool);
        assert_eq!(
            bool_tensor.data.into_raw_view().expect("raw view"),
            vec![0, 1]
        );
    }

    #[test]
    fn constructors_validate_shape_product() {
        let err = WireTensor::from_f32(vec![2, 2], vec![1.0, 2.0]).unwrap_err();
        assert!(err.contains("does not match shape"));
    }

    #[test]
    fn resolve_aux_frame_shares_bytes_using_one_based_message_index() {
        let mut tensor = WireTensor {
            dtype: TensorDtype::F32,
            shape: vec![2],
            data: WireArrayData::AuxIndex(1),
        };
        let bytes = Bytes::from_static(b"payload");
        let bytes_ptr = bytes.as_ptr();

        tensor.resolve_aux_frame(&[Bytes::new(), bytes]).unwrap();

        let resolved = tensor.data.as_raw_view().unwrap();
        assert_eq!(resolved.as_ptr(), bytes_ptr);
        assert_eq!(resolved.as_ref(), b"payload");
    }
}
