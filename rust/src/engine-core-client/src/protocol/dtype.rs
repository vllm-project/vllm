// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};

/// Element types accepted by Python's tensor decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeDisplay, DeserializeFromStr)]
pub enum TensorDtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    Bf16,
    F32,
    F64,
}

impl TensorDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::I8 => "int8",
            Self::U16 => "uint16",
            Self::I16 => "int16",
            Self::U32 => "uint32",
            Self::I32 => "int32",
            Self::U64 => "uint64",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::Bf16 => "bfloat16",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for TensorDtype {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bool" => Ok(Self::Bool),
            "uint8" => Ok(Self::U8),
            "int8" => Ok(Self::I8),
            "uint16" => Ok(Self::U16),
            "int16" => Ok(Self::I16),
            "uint32" => Ok(Self::U32),
            "int32" => Ok(Self::I32),
            "uint64" => Ok(Self::U64),
            "int64" => Ok(Self::I64),
            "float16" => Ok(Self::F16),
            "bfloat16" => Ok(Self::Bf16),
            "float32" => Ok(Self::F32),
            "float64" => Ok(Self::F64),
            _ => Err(format!("unsupported tensor dtype {value:?}")),
        }
    }
}

/// Byte order carried by a NumPy dtype descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
    Native,
}

/// Numeric array dtype with an explicit byte-order contract.
///
/// Also accepts Torch type names as native-endian dtypes because prompt
/// logprobs use tensor tuples with the same wire shape as NumPy arrays.
/// Serializes byte order explicitly, including `=` for native byte order.
/// Bfloat16 uses the Torch name after the prefix as a nonstandard descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, SerializeDisplay, DeserializeFromStr)]
pub struct NumpyDtype {
    pub scalar: TensorDtype,
    pub endianness: Endianness,
}

impl NumpyDtype {
    pub const fn little(scalar: TensorDtype) -> Self {
        Self {
            scalar,
            endianness: Endianness::Little,
        }
    }

    pub const fn native(scalar: TensorDtype) -> Self {
        Self {
            scalar,
            endianness: Endianness::Native,
        }
    }

    pub const fn big(scalar: TensorDtype) -> Self {
        Self {
            scalar,
            endianness: Endianness::Big,
        }
    }
}

impl From<TensorDtype> for NumpyDtype {
    fn from(scalar: TensorDtype) -> Self {
        Self::native(scalar)
    }
}

impl fmt::Display for NumpyDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = match self.endianness {
            Endianness::Native => '=',
            Endianness::Little => '<',
            Endianness::Big => '>',
        };
        let code = match self.scalar {
            TensorDtype::Bool => "b1",
            TensorDtype::U8 => "u1",
            TensorDtype::I8 => "i1",
            TensorDtype::U16 => "u2",
            TensorDtype::I16 => "i2",
            TensorDtype::U32 => "u4",
            TensorDtype::I32 => "i4",
            TensorDtype::U64 => "u8",
            TensorDtype::I64 => "i8",
            TensorDtype::F16 => "f2",
            TensorDtype::F32 => "f4",
            TensorDtype::F64 => "f8",
            TensorDtype::Bf16 => "bfloat16",
        };
        write!(f, "{prefix}{code}")
    }
}

impl FromStr for NumpyDtype {
    type Err = String;

    fn from_str(dtype: &str) -> Result<Self, Self::Err> {
        let (endianness, body) = match dtype.as_bytes().first().copied() {
            Some(b'<') => (Endianness::Little, &dtype[1..]),
            Some(b'>') => (Endianness::Big, &dtype[1..]),
            Some(b'=' | b'|') => (Endianness::Native, &dtype[1..]),
            _ => (Endianness::Native, dtype),
        };
        let scalar = match body {
            "b1" => TensorDtype::Bool,
            "u1" => TensorDtype::U8,
            "i1" => TensorDtype::I8,
            "u2" => TensorDtype::U16,
            "i2" => TensorDtype::I16,
            "u4" => TensorDtype::U32,
            "i4" => TensorDtype::I32,
            "u8" => TensorDtype::U64,
            "i8" => TensorDtype::I64,
            "f2" => TensorDtype::F16,
            "f4" => TensorDtype::F32,
            "f8" => TensorDtype::F64,
            _ => body.parse().map_err(|_| format!("unsupported dtype string {dtype:?}"))?,
        };
        Ok(Self { scalar, endianness })
    }
}

/// Effective model dtype reported by the engine after config resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Endianness, ModelDtype, NumpyDtype, TensorDtype};

    #[test]
    fn numpy_descriptors_preserve_explicit_byte_order() {
        for (code, scalar) in [
            ("b1", TensorDtype::Bool),
            ("u1", TensorDtype::U8),
            ("i1", TensorDtype::I8),
            ("u2", TensorDtype::U16),
            ("i2", TensorDtype::I16),
            ("u4", TensorDtype::U32),
            ("i4", TensorDtype::I32),
            ("u8", TensorDtype::U64),
            ("i8", TensorDtype::I64),
            ("f2", TensorDtype::F16),
            ("bfloat16", TensorDtype::Bf16),
            ("f4", TensorDtype::F32),
            ("f8", TensorDtype::F64),
        ] {
            for (prefix, endianness) in [("<", Endianness::Little), (">", Endianness::Big)] {
                let wire = serde_json::json!(format!("{prefix}{code}"));
                let dtype: NumpyDtype = serde_json::from_value(wire.clone()).unwrap();
                assert_eq!(dtype, NumpyDtype { scalar, endianness });
                assert_eq!(serde_json::to_value(dtype).unwrap(), wire);
            }
            for wire in [
                format!("={code}"),
                format!("|{code}"),
                code.into(),
                scalar.as_str().into(),
            ] {
                let dtype: NumpyDtype = serde_json::from_value(serde_json::json!(wire)).unwrap();
                assert_eq!(dtype, NumpyDtype::from(scalar));
                assert_eq!(
                    serde_json::to_value(dtype).unwrap(),
                    serde_json::json!(format!("={code}"))
                );
            }
        }
    }

    #[test]
    fn rejects_unsupported_numpy_dtypes() {
        for wire in ["", "object", "<c8", "<bf2"] {
            assert!(serde_json::from_value::<NumpyDtype>(serde_json::json!(wire)).is_err());
        }
    }

    #[test]
    fn serde_uses_protocol_dtype_strings() {
        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert_eq!(
            serde_json::from_value::<ModelDtype>(serde_json::json!("bfloat16")).unwrap(),
            ModelDtype::BFloat16
        );
        assert_eq!(ModelDtype::Float32.as_str(), "float32");
    }
}
