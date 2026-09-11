// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// One worker rank's encoded KV connector handshake metadata, as returned by
/// the `get_kv_connector_handshake_entries` utility call.
///
/// `payload` is the connector's own msgpack encoding of its handshake
/// metadata and stays opaque to the frontend.
///
/// Original Python definition (`KVConnectorHandshakeEntry`) lives in
/// `vllm/distributed/kv_transfer/kv_connector/v1/base.py`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvConnectorHandshakeEntry {
    pub pp_rank: u32,
    pub tp_rank: u32,
    pub payload: Bytes,
    #[serde(default)]
    pub compatibility_hash: Option<String>,
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;

    fn entry_value(payload: Value) -> Value {
        Value::Map(vec![
            (Value::from("pp_rank"), Value::from(1)),
            (Value::from("tp_rank"), Value::from(2)),
            (Value::from("payload"), payload),
            (Value::from("compatibility_hash"), Value::from("hash")),
        ])
    }

    #[test]
    fn decodes_payload_from_rmpv_binary_and_from_slice() {
        let expected = KvConnectorHandshakeEntry {
            pp_rank: 1,
            tp_rank: 2,
            payload: Bytes::from_static(&[0xde, 0xad]),
            compatibility_hash: Some("hash".to_string()),
        };

        let from_value: KvConnectorHandshakeEntry =
            rmpv::ext::from_value(entry_value(Value::Binary(vec![0xde, 0xad]))).unwrap();
        assert_eq!(from_value, expected);

        let encoded = rmp_serde::to_vec_named(&expected).unwrap();
        let from_slice: KvConnectorHandshakeEntry = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(from_slice, expected);
    }

    #[test]
    fn missing_compatibility_hash_decodes_as_none() {
        let value = Value::Map(vec![
            (Value::from("pp_rank"), Value::from(0)),
            (Value::from("tp_rank"), Value::from(0)),
            (Value::from("payload"), Value::Binary(vec![])),
        ]);
        let entry: KvConnectorHandshakeEntry = rmpv::ext::from_value(value).unwrap();
        assert_eq!(entry.compatibility_hash, None);
        assert!(entry.payload.is_empty());
    }
}
