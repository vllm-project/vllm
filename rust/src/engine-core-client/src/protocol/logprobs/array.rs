// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::io::Cursor;

use byteorder::{BigEndian, LittleEndian, NativeEndian, ReadBytesExt};
use bytes::Bytes;
use itertools::Itertools as _;

use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::dtype::{Endianness, TensorDtype};
use crate::protocol::tensor::{ShapeExt as _, WireNdArray};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DecodedArray2<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

pub(super) fn decode_array2_u32(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
) -> Result<DecodedArray2<u32>> {
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, &[TensorDtype::I32, TensorDtype::I64])?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }

    let data = match scalar {
        TensorDtype::I32 => decode_i32_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        TensorDtype::I64 => decode_i64_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        _ => unreachable!("scalar validation should accept only i32 and i64"),
    };
    Ok(DecodedArray2 {
        rows: shape[0],
        cols: shape[1],
        data,
    })
}

pub(super) fn decode_array1_u32(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
) -> Result<Vec<u32>> {
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, &[TensorDtype::I32, TensorDtype::I64])?;
    if shape.len() != 1 {
        return Err(decode_error(
            field,
            &format!("expected rank-1 array, got rank {}", shape.len()),
        ));
    }

    let data = match scalar {
        TensorDtype::I32 => decode_i32_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        TensorDtype::I64 => decode_i64_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        _ => unreachable!("scalar validation should accept only i32 and i64"),
    };
    Ok(data)
}

pub(super) fn decode_array2_f32(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
) -> Result<DecodedArray2<f32>> {
    let (shape, bytes, _, endianness) =
        decode_array_metadata(value, field, frames, &[TensorDtype::F32])?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }

    let data = decode_f32_vec(&bytes, endianness, field)?;
    Ok(DecodedArray2 {
        rows: shape[0],
        cols: shape[1],
        data,
    })
}

pub(super) fn decode_array_metadata(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
    expected_scalars: &[TensorDtype],
) -> Result<(Vec<usize>, Bytes, TensorDtype, Endianness)> {
    let mut value = value;
    let scalar = value.dtype.scalar;
    let endianness = value.dtype.endianness;
    if !expected_scalars.contains(&scalar) {
        return Err(decode_error(
            field,
            &format!(
                "expected dtype in {:?}, got {:?}",
                expected_scalars, value.dtype
            ),
        ));
    }

    value
        .resolve_aux_frame(frames)
        .map_err(|message| decode_error(field, &message))?;
    let WireNdArray { shape, data, .. } = value;
    let bytes = data.into_raw_view().expect("auxiliary frame reference was resolved above");
    validate_byte_length(shape.as_slice(), bytes.len(), field, scalar)?;
    Ok((shape, bytes, scalar, endianness))
}

pub(super) fn validate_byte_length(
    shape: &[usize],
    byte_len: usize,
    field: &str,
    scalar: TensorDtype,
) -> Result<()> {
    let element_count = shape
        .checked_numel()
        .ok_or_else(|| decode_error(field, "shape element count overflowed usize"))?;
    let element_size = scalar.element_size();
    let expected = element_count
        .checked_mul(element_size)
        .ok_or_else(|| decode_error(field, "byte length overflowed usize"))?;
    if expected != byte_len {
        return Err(decode_error(
            field,
            &format!("byte length mismatch: expected {expected}, got {byte_len}"),
        ));
    }
    Ok(())
}

pub(super) fn decode_i32_vec(
    bytes: &[u8],
    endianness: Endianness,
    field: &str,
) -> Result<Vec<i32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by 4", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / 4);
    while (cursor.position() as usize) < bytes.len() {
        let value = match endianness {
            Endianness::Little => cursor.read_i32::<LittleEndian>(),
            Endianness::Big => cursor.read_i32::<BigEndian>(),
            Endianness::Native => cursor.read_i32::<NativeEndian>(),
        }
        .map_err(|error| decode_error(field, &format!("failed to read i32 payload: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

pub(super) fn decode_f32_vec(
    bytes: &[u8],
    endianness: Endianness,
    field: &str,
) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by 4", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / 4);
    while (cursor.position() as usize) < bytes.len() {
        let value = match endianness {
            Endianness::Little => cursor.read_f32::<LittleEndian>(),
            Endianness::Big => cursor.read_f32::<BigEndian>(),
            Endianness::Native => cursor.read_f32::<NativeEndian>(),
        }
        .map_err(|error| decode_error(field, &format!("failed to read f32 payload: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

pub(super) fn decode_i64_vec(
    bytes: &[u8],
    endianness: Endianness,
    field: &str,
) -> Result<Vec<i64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by 8", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / 8);
    while (cursor.position() as usize) < bytes.len() {
        let value = match endianness {
            Endianness::Little => cursor.read_i64::<LittleEndian>(),
            Endianness::Big => cursor.read_i64::<BigEndian>(),
            Endianness::Native => cursor.read_i64::<NativeEndian>(),
        }
        .map_err(|error| decode_error(field, &format!("failed to read i64 payload: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

fn convert_to_u32<I>(value: I, field: &str) -> Result<u32>
where
    I: TryInto<u32> + std::fmt::Display + Copy,
{
    value.try_into().map_err(|_| {
        decode_error(
            field,
            &format!("expected non-negative token id/rank that fits in u32, got {value}"),
        )
    })
}

pub(super) fn decode_error(field: &str, reason: &str) -> Error {
    ext_value_decode!("{field}: {reason}")
}
