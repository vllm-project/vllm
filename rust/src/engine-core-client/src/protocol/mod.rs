use std::any::type_name;
use std::io::Cursor;

use rmpv::Value;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport;

use crate::error::{Error, Result};

/// Dynamic msgpack value used for schema positions that are preserved but not
/// yet strongly typed in the early-stage Rust client.
pub type OpaqueValue = Value;

fn default_opaque_value_nil() -> OpaqueValue {
    Value::Nil
}

fn is_false(v: &bool) -> bool {
    !v
}

pub mod dtype;
pub mod handshake;
pub mod logprobs;
pub mod lora;
pub mod multimodal;
pub mod output;
pub mod request;
pub mod sampling;
pub mod stats;
pub mod structured_outputs;
pub mod tensor;
pub mod utility;

/// Encode a Rust value into msgpack using the protocol crate's serde model.
pub fn encode_msgpack<T>(value: &T) -> Result<Vec<u8>>
where
    T: Serialize + std::fmt::Debug,
{
    rmp_serde::to_vec_named(value).map_err(|error| Error::Encode {
        target_type: type_name::<T>(),
        message: format!(
            "failed to encode value `{:?}`: {}",
            value,
            error.to_report_string()
        ),
    })
}

/// Decode a msgpack payload into a strongly typed protocol value, with enhanced
/// error reporting.
pub fn decode_msgpack<T>(bytes: &[u8]) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    fn decode_value_preview(bytes: &[u8]) -> String {
        match decode_value(bytes) {
            Ok(value) => format!("{value}"),
            Err(error) => format!("<value decode failed: {error}>"),
        }
    }

    rmp_serde::from_slice(bytes).map_err(|error| Error::Decode {
        target_type: type_name::<T>(),
        message: format!("{error}; value fallback: {}", decode_value_preview(bytes)),
    })
}

/// Decode a msgpack payload into a dynamic value for diagnostics and tests.
pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    Ok(rmpv::decode::read_value(&mut Cursor::new(bytes))?)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn decode_msgpack_includes_type_name_and_value_fallback() {
        let error = decode_msgpack::<u64>(
            &rmp_serde::to_vec_named(&BTreeMap::from([("status", "READY")])).unwrap(),
        )
        .unwrap_err();

        expect_test::expect![[r#"messagepack decode failed for u64: wrong msgpack marker FixMap(1); value fallback: {"status": "READY"}"#]].assert_eq(&error.to_report_string());
    }
}
