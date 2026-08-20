// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, Result, bail_ext_value_decode, ext_value_decode};
use crate::protocol::tensor::{ShapeExt as _, WireArrayData, WireNdArray};

/// Semantic routed-experts payload returned by engine-core.
///
/// The first dimension is the token dimension; the remaining dimensions are
/// model layers and experts selected per token. Engine-core emits the compact
/// unsigned dtype chosen by its scheduler-side routed-experts buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutedExperts {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl RoutedExperts {
    pub fn append(&mut self, mut other: Self) -> std::result::Result<(), String> {
        if self.dtype != other.dtype {
            return Err(format!(
                "routed-experts dtype changed across output chunks: {} != {}",
                self.dtype, other.dtype
            ));
        }
        if self.shape.len() != 3 || other.shape.len() != 3 || self.shape[1..] != other.shape[1..] {
            return Err(format!(
                "routed-experts trailing shape changed across output chunks: {:?} != {:?}",
                self.shape, other.shape
            ));
        }
        self.shape[0] = self.shape[0]
            .checked_add(other.shape[0])
            .ok_or_else(|| "routed-experts token dimension overflowed usize".to_string())?;
        self.data.append(&mut other.data);
        Ok(())
    }

    /// Serialize the tensor using NumPy's version 1.0 `.npy` format.
    pub fn to_npy_bytes(&self) -> Result<Vec<u8>> {
        if self.shape.len() != 3 {
            bail_ext_value_decode!(
                "routed_experts: expected a rank-3 ndarray, got shape {:?}",
                self.shape
            );
        }
        let (descriptor, item_size) = match self.dtype.as_str() {
            "uint8" => ("|u1", 1usize),
            "uint16" => ("<u2", 2usize),
            other => bail_ext_value_decode!(
                "routed_experts: expected normalized uint8 or uint16 dtype, got {other:?}"
            ),
        };
        let expected = self
            .shape
            .checked_numel()
            .and_then(|numel| numel.checked_mul(item_size))
            .ok_or_else(|| {
                ext_value_decode!(
                    "routed_experts: shape byte length overflowed usize: {:?}",
                    self.shape
                )
            })?;
        if self.data.len() != expected {
            bail_ext_value_decode!(
                "routed_experts: byte length mismatch: expected {expected}, got {}",
                self.data.len()
            );
        }

        let dictionary = format!(
            "{{'descr': '{descriptor}', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
            self.shape[0], self.shape[1], self.shape[2]
        );
        const PREAMBLE_LEN: usize = 10;
        const ARRAY_ALIGNMENT: usize = 64;
        let padding = ARRAY_ALIGNMENT - ((PREAMBLE_LEN + dictionary.len() + 1) % ARRAY_ALIGNMENT);
        let header_len = dictionary
            .len()
            .checked_add(padding)
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| ext_value_decode!("routed_experts: NumPy header length overflow"))?;
        let header_len = u16::try_from(header_len).map_err(|_| {
            ext_value_decode!("routed_experts: NumPy v1 header does not fit in uint16")
        })?;

        let mut encoded = Vec::with_capacity(PREAMBLE_LEN + usize::from(header_len) + expected);
        encoded.extend_from_slice(b"\x93NUMPY\x01\x00");
        encoded.extend_from_slice(&header_len.to_le_bytes());
        encoded.extend_from_slice(dictionary.as_bytes());
        encoded.resize(encoded.len() + padding, b' ');
        encoded.push(b'\n');
        encoded.extend_from_slice(&self.data);
        Ok(encoded)
    }
}

/// Routed-experts output is initially decoded from Python's ndarray wire
/// tuple and resolved against optional multipart frames before it is exposed.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum MaybeWireRoutedExperts {
    Wire(WireNdArray),
    Direct(RoutedExperts),
}

impl<'de> Deserialize<'de> for MaybeWireRoutedExperts {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireNdArray::deserialize(deserializer).map(Self::Wire)
    }
}

impl Serialize for MaybeWireRoutedExperts {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(value) => {
                WireNdArray::from_raw(value.dtype.clone(), value.shape.clone(), value.data.clone())
                    .serialize(serializer)
            }
        }
    }
}

impl MaybeWireRoutedExperts {
    pub(super) fn resolve<Frame>(self, frames: &[Frame]) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        let Self::Wire(WireNdArray { dtype, shape, data }) = self else {
            return Ok(self);
        };
        if shape.len() != 3 {
            bail_ext_value_decode!(
                "routed_experts: expected a rank-3 ndarray, got shape {shape:?}"
            );
        }
        let (dtype, item_size) = match dtype.as_str() {
            "|u1" | "uint8" => ("uint8", 1usize),
            "<u2" | "=u2" | "uint16" => ("uint16", 2usize),
            other => bail_ext_value_decode!(
                "routed_experts: expected native uint8 or uint16 dtype, got {other:?}"
            ),
        };
        let data = match data {
            WireArrayData::RawView(bytes) => bytes,
            WireArrayData::AuxIndex(index) => {
                let frame = frames.get(index).ok_or_else(|| {
                    ext_value_decode!(
                        "routed_experts: aux frame index {index} out of range for {} frames",
                        frames.len()
                    )
                })?;
                Bytes::copy_from_slice(frame.as_ref())
            }
        };
        let expected = shape
            .checked_numel()
            .and_then(|numel| numel.checked_mul(item_size))
            .ok_or_else(|| {
                ext_value_decode!("routed_experts: shape byte length overflowed usize: {shape:?}")
            })?;
        if data.len() != expected {
            bail_ext_value_decode!(
                "routed_experts: byte length mismatch: expected {expected}, got {}",
                data.len()
            );
        }
        Ok(Self::Direct(RoutedExperts {
            dtype: dtype.to_string(),
            shape,
            data: data.to_vec(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_multipart_uint16_routed_experts() {
        let payload = [1_u16, 2, 300, 4].into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        let frames = [Bytes::new(), Bytes::from(payload.clone())];
        let decoded = MaybeWireRoutedExperts::Wire(WireNdArray {
            dtype: "<u2".to_string(),
            shape: vec![2, 1, 2],
            data: WireArrayData::AuxIndex(1),
        })
        .resolve(&frames)
        .expect("resolve routed experts")
        .into_direct()
        .expect("direct routed experts");

        assert_eq!(decoded.dtype, "uint16");
        assert_eq!(decoded.shape, [2, 1, 2]);
        assert_eq!(decoded.data, payload);
    }

    #[test]
    fn rejects_non_rank_three_routed_experts() {
        let error = MaybeWireRoutedExperts::Wire(WireNdArray {
            dtype: "|u1".to_string(),
            shape: vec![2, 2],
            data: WireArrayData::RawView(Bytes::from_static(&[1, 2, 3, 4])),
        })
        .resolve(&[Bytes::new()])
        .expect_err("rank must be checked");

        assert!(error.to_string().contains("expected a rank-3 ndarray"));
    }

    #[test]
    fn serializes_uint8_as_numpy_v1() {
        let routed = RoutedExperts {
            dtype: "uint8".to_string(),
            shape: vec![2, 1, 2],
            data: vec![1, 2, 3, 4],
        };

        assert_eq!(
            routed.to_npy_bytes().expect("serialize routed experts"),
            b"\x93NUMPY\x01\x00\x76\x00{'descr': '|u1', 'fortran_order': False, 'shape': (2, 1, 2), }                                                       \n\x01\x02\x03\x04"
        );
    }

    #[test]
    fn serializes_uint16_as_numpy_v1() {
        let routed = RoutedExperts {
            dtype: "uint16".to_string(),
            shape: vec![2, 1, 2],
            data: vec![1, 0, 2, 0, 3, 0, 4, 0],
        };

        assert_eq!(
            routed.to_npy_bytes().expect("serialize routed experts"),
            b"\x93NUMPY\x01\x00\x76\x00{'descr': '<u2', 'fortran_order': False, 'shape': (2, 1, 2), }                                                       \n\x01\x00\x02\x00\x03\x00\x04\x00"
        );
    }
}
