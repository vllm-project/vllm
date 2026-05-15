use std::io::Cursor;

use byteorder::{BigEndian, LittleEndian, NativeEndian, ReadBytesExt};
use itertools::Itertools as _;

use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::tensor_wire::{WireArrayData, WireNdArray};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarType {
    I32,
    I64,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Endianness {
    Little,
    Big,
    Native,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DecodedArray2<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

pub(super) fn decode_array2_u32<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
) -> Result<DecodedArray2<u32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, &[ScalarType::I32, ScalarType::I64])?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }

    let data = match scalar {
        ScalarType::I32 => decode_i32_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        ScalarType::I64 => decode_i64_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        ScalarType::F32 => unreachable!("scalar validation should reject f32"),
    };
    Ok(DecodedArray2 {
        rows: shape[0],
        cols: shape[1],
        data,
    })
}

pub(super) fn decode_array1_u32<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
) -> Result<Vec<u32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, &[ScalarType::I32, ScalarType::I64])?;
    if shape.len() != 1 {
        return Err(decode_error(
            field,
            &format!("expected rank-1 array, got rank {}", shape.len()),
        ));
    }

    let data = match scalar {
        ScalarType::I32 => decode_i32_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        ScalarType::I64 => decode_i64_vec(&bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .try_collect()?,
        ScalarType::F32 => unreachable!("scalar validation should reject f32"),
    };
    Ok(data)
}

pub(super) fn decode_array2_f32<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
) -> Result<DecodedArray2<f32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, _, endianness) =
        decode_array_metadata(value, field, frames, &[ScalarType::F32])?;
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

pub(super) fn decode_array_metadata<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
    expected_scalars: &[ScalarType],
) -> Result<(Vec<usize>, Vec<u8>, ScalarType, Endianness)>
where
    Frame: AsRef<[u8]>,
{
    let WireNdArray { dtype, shape, data } = value;
    let (scalar, endianness) = parse_dtype(&dtype, field)?;
    if !expected_scalars.contains(&scalar) {
        return Err(decode_error(
            field,
            &format!("expected dtype in {:?}, got {}", expected_scalars, dtype),
        ));
    }

    let bytes = resolve_array_bytes(data, field, frames)?;
    validate_byte_length(shape.as_slice(), bytes.len(), field, scalar)?;
    Ok((shape, bytes, scalar, endianness))
}

pub(super) fn parse_dtype(dtype: &str, field: &str) -> Result<(ScalarType, Endianness)> {
    let (endianness, body) = match dtype.as_bytes().first().copied() {
        Some(b'<') => (Endianness::Little, &dtype[1..]),
        Some(b'>') => (Endianness::Big, &dtype[1..]),
        Some(b'=') => (Endianness::Native, &dtype[1..]),
        Some(b'|') => (Endianness::Native, &dtype[1..]),
        _ => (Endianness::Native, dtype),
    };

    let scalar = match body {
        "i4" | "int32" => ScalarType::I32,
        "i8" | "int64" => ScalarType::I64,
        "f4" | "float32" => ScalarType::F32,
        _ => {
            return Err(decode_error(
                field,
                &format!("unsupported dtype string {dtype:?}"),
            ));
        }
    };
    Ok((scalar, endianness))
}

pub(super) fn resolve_array_bytes<Frame>(
    value: WireArrayData,
    field: &str,
    frames: &[Frame],
) -> Result<Vec<u8>>
where
    Frame: AsRef<[u8]>,
{
    match value {
        WireArrayData::RawView(bytes) => Ok(bytes),
        WireArrayData::AuxIndex(index) => {
            let frame = frames.get(index).ok_or_else(|| {
                decode_error(
                    field,
                    &format!(
                        "aux frame index {index} out of range for {} frames",
                        frames.len()
                    ),
                )
            })?;
            Ok(frame.as_ref().to_vec())
        }
    }
}

pub(super) fn validate_byte_length(
    shape: &[usize],
    byte_len: usize,
    field: &str,
    scalar: ScalarType,
) -> Result<()> {
    let element_count = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| decode_error(field, "shape element count overflowed usize"))?;
    let element_size = match scalar {
        ScalarType::I32 | ScalarType::F32 => 4,
        ScalarType::I64 => 8,
    };
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
