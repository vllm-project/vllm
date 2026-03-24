use std::io::Cursor;

use byteorder::{BigEndian, LittleEndian, NativeEndian, ReadBytesExt};
use ndarray::{Array1, Array2};

use crate::error::{Error, Result};
use crate::protocol::aux::wire::{WireArrayData, WireNdArray};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarType {
    I32,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Endianness {
    Little,
    Big,
    Native,
}

pub(super) fn decode_array2_i32<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
) -> Result<Array2<i32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, endianness) = decode_array_metadata(value, field, frames, ScalarType::I32)?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }

    let data = decode_i32_vec(&bytes, endianness, field)?;
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|error| decode_error(field, &format!("invalid shape: {error}")))
}

pub(super) fn decode_array1_i32<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
) -> Result<Array1<i32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, endianness) = decode_array_metadata(value, field, frames, ScalarType::I32)?;
    if shape.len() != 1 {
        return Err(decode_error(
            field,
            &format!("expected rank-1 array, got rank {}", shape.len()),
        ));
    }

    Ok(Array1::from_vec(decode_i32_vec(&bytes, endianness, field)?))
}

pub(super) fn decode_array2_f32<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
) -> Result<Array2<f32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, endianness) = decode_array_metadata(value, field, frames, ScalarType::F32)?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }

    let data = decode_f32_vec(&bytes, endianness, field)?;
    Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|error| decode_error(field, &format!("invalid shape: {error}")))
}

pub(super) fn decode_array_metadata<Frame>(
    value: WireNdArray,
    field: &str,
    frames: &[Frame],
    expected_scalar: ScalarType,
) -> Result<(Vec<usize>, Vec<u8>, Endianness)>
where
    Frame: AsRef<[u8]>,
{
    let WireNdArray { dtype, shape, data } = value;
    let (scalar, endianness) = parse_dtype(&dtype, field)?;
    if scalar != expected_scalar {
        return Err(decode_error(
            field,
            &format!("expected dtype {:?}, got {}", expected_scalar, dtype),
        ));
    }

    let bytes = resolve_array_bytes(data, field, frames)?;
    validate_byte_length(shape.as_slice(), bytes.len(), field, expected_scalar)?;
    Ok((shape, bytes, endianness))
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

pub(super) fn decode_error(field: &str, reason: &str) -> Error {
    Error::ValueDecodeExt(format!("{field}: {reason}"))
}
