// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Wire forms of prepared multimodal items shared by the HTTP `features` and
//! gRPC `PreprocessedMediaFeatures` formats. Kwargs use the engine protocol's
//! MessagePack encoding, which Python's `MsgpackDecoder` also reads.

use std::io::Cursor;

use serde::Deserialize as _;
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::encode_msgpack;
use vllm_engine_core_client::protocol::multimodal::MmKwargsItem;
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};

const MAX_MSGPACK_DEPTH: usize = 32;

/// Encode one item with all tensors inline.
pub(crate) fn encode_kwargs(item: &MmKwargsItem) -> Result<Vec<u8>, String> {
    encode_msgpack(item).map_err(|error| error.to_report_string())
}

/// Decode one item using the engine protocol's typed serde schema, rejecting
/// deep nesting and trailing data.
pub(crate) fn decode_kwargs(bytes: &[u8]) -> Result<MmKwargsItem, String> {
    let mut decoder = rmp_serde::Deserializer::new(Cursor::new(bytes));
    decoder.set_max_depth(MAX_MSGPACK_DEPTH);
    let kwargs = MmKwargsItem::deserialize(&mut decoder).map_err(|error| {
        format!(
            "invalid multimodal kwargs MessagePack: {}",
            error.as_report()
        )
    })?;
    if decoder.position() != bytes.len() as u64 {
        return Err("kwargs contains trailing MessagePack data".to_string());
    }
    Ok(kwargs)
}

/// Read an inline boolean `is_embed` mask.
pub(crate) fn mask_to_bools(mask: &WireTensor) -> Result<Vec<bool>, String> {
    match &mask.data {
        WireArrayData::RawView(bytes) => Ok(bytes.iter().map(|&byte| byte != 0).collect()),
        WireArrayData::AuxIndex(_) => Err("is_embed mask must be stored inline".to_string()),
    }
}
