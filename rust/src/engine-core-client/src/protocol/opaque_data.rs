// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;

use bytes::Bytes;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Format-agnostic binary data produced by engine-core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueData(Bytes);

impl OpaqueData {
    pub fn new(data: impl Into<Bytes>) -> Self {
        Self(data.into())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<u8> {
        self.0.to_vec()
    }
}

impl Serialize for OpaqueData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for OpaqueData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct OpaqueDataVisitor;

        impl<'de> Visitor<'de> for OpaqueDataVisitor {
            type Value = OpaqueData;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("opaque binary data")
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(OpaqueData::new(Bytes::copy_from_slice(value)))
            }

            fn visit_borrowed_bytes<E>(self, value: &'de [u8]) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(OpaqueData::new(Bytes::copy_from_slice(value)))
            }

            fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(OpaqueData::new(value))
            }
        }

        deserializer.deserialize_bytes(OpaqueDataVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    #[test]
    fn messagepack_round_trip_preserves_bytes() {
        let expected = OpaqueData::new(vec![0, 1, 2, 0xff]);
        let encoded = encode_msgpack(&expected).expect("encode opaque data");
        let decoded: OpaqueData = decode_msgpack(&encoded).expect("decode opaque data");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn messagepack_round_trip_preserves_large_bytes() {
        let expected = OpaqueData::new(vec![0x5a; 6 * 1024 * 1024 + 1]);
        let encoded = encode_msgpack(&expected).expect("encode large opaque data");
        let decoded: OpaqueData = decode_msgpack(&encoded).expect("decode large opaque data");
        assert_eq!(decoded, expected);
    }
}
