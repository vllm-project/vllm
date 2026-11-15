// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Writer-side serialization for the shm object storage.
//!
//! Each `MmKwargsItem` is stored as a single raw msgpack body with all tensor
//! buffers inline (`WireArrayData::RawView` serializes as ext type 3), prefixed
//! by the [`MM_ITEM_FORMAT_MARKER`] byte. The Python receiver detects the
//! marker in `MsgpackSerde.deserialize` and decodes the body directly with
//! `MsgpackDecoder(MultiModalKwargsItem, share_mem=False)`; only the serialize
//! direction is needed here.

use crate::error::{Error, Result, serialize_msgpack};
use vllm_engine_core_client::protocol::multimodal::MmKwargsItem;

/// Format marker prefixing the msgpack body, detected by
/// `MsgpackSerde.deserialize`. Pickle protocol >= 2 streams always start with
/// `\x80` (PROTO), so 0x01 is unambiguous against the legacy format.
pub(crate) const MM_ITEM_FORMAT_MARKER: u8 = 0x01;

/// An `io::Write` that only counts bytes, used to size a shm chunk before
/// serializing into it (no allocation or copying of tensor bytes).
struct CountWriter(usize);

impl std::io::Write for CountWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
    }

    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.0 += buf.len();
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Exact serialized length of `[0x01][msgpack body]`, computed by a counting
/// write pass. Unlike the ZMQ client path, tensors always stay inline in the
/// body, so no auxiliary frames are needed.
pub(crate) fn serialized_mm_item_len(item: &MmKwargsItem) -> Result<usize> {
    let mut counter = CountWriter(0);
    rmp_serde::encode::write_named(&mut counter, item)
        .map_err(|error| serialize_msgpack!("{error}"))?;
    Ok(1 + counter.0)
}

/// Serialize an item as `[0x01][msgpack body]` into `buf`, which must be
/// exactly `serialized_mm_item_len(item)` bytes.
pub(crate) fn serialize_mm_item_into(item: &MmKwargsItem, buf: &mut [u8]) -> Result<()> {
    buf[0] = MM_ITEM_FORMAT_MARKER;
    let mut writer: &mut [u8] = &mut buf[1..];
    rmp_serde::encode::write_named(&mut writer, item)
        .map_err(|error| serialize_msgpack!("{error}"))?;
    debug_assert!(writer.is_empty(), "counting pass sized the buffer exactly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmField, MmFieldElem, MmKwargValue,
    };
    use vllm_engine_core_client::protocol::tensor::WireTensor;

    fn int_elem(value: i64) -> MmFieldElem {
        MmFieldElem {
            data: Some(MmKwargValue::Int(value)),
            field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
        }
    }

    #[test]
    fn serialize_writes_marker_and_inline_msgpack_body() {
        let big_data = (0..128).map(|value| value as f32).collect::<Vec<_>>();
        let big_bytes = big_data.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>();
        let small_bytes = [1.0_f32, 2.0].into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>();
        let item: MmKwargsItem = BTreeMap::from([
            ("offset".to_string(), int_elem(7)),
            (
                "pixel_values".to_string(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_f32(vec![128], big_data).unwrap(),
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                },
            ),
            (
                "small".to_string(),
                MmFieldElem {
                    data: Some(MmKwargValue::Tensor(
                        WireTensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap(),
                    )),
                    field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                },
            ),
        ]);

        let mut payload = vec![0u8; serialized_mm_item_len(&item).unwrap()];
        serialize_mm_item_into(&item, &mut payload).unwrap();
        assert_eq!(payload[0], MM_ITEM_FORMAT_MARKER);

        // Both tensors stay inline as raw-view exts regardless of size.
        let body = vllm_engine_core_client::protocol::decode_value(&payload[1..]).unwrap();
        let map = body.as_map().expect("item body is a map");
        let field_value = |key: &str| {
            map.iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .unwrap_or_else(|| panic!("missing key {key}"))
                .1
                .clone()
        };
        let tensor_data = |key: &str| {
            let elem = field_value(key);
            let data = elem
                .as_map()
                .unwrap()
                .iter()
                .find(|(k, _)| k.as_str() == Some("data"))
                .map(|(_, v)| v.clone())
                .expect("data field");
            let tensor_tuple = data.as_array().expect("tensor tuple");
            tensor_tuple[2].clone()
        };
        assert_eq!(tensor_data("pixel_values"), rmpv::Value::Ext(3, big_bytes));
        assert_eq!(tensor_data("small"), rmpv::Value::Ext(3, small_bytes));

        let offset = field_value("offset");
        let offset_data = offset
            .as_map()
            .unwrap()
            .iter()
            .find(|(k, _)| k.as_str() == Some("data"))
            .map(|(_, v)| v)
            .expect("data field");
        assert_eq!(offset_data.as_i64(), Some(7));
    }
}
