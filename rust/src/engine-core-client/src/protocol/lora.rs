use std::any::type_name;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror_ext::AsReport;

use crate::error::{Error, Result};
use crate::protocol::OpaqueValue;

/// Request for a LoRA adapter.
///
/// Mirrors Python `vllm.lora.request.LoRARequest`, which is a msgspec
/// `array_like=True` struct. Keep the field order aligned with Python.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct LoRARequest {
    pub lora_name: String,
    pub lora_int_id: u64,
    pub lora_path: String,
    #[serde(default)]
    pub base_model_name: Option<String>,
    #[serde(default)]
    pub tensorizer_config_dict: Option<OpaqueValue>,
    #[serde(default)]
    pub load_inplace: bool,
    #[serde(default)]
    pub is_3d_lora_weight: bool,
}

impl LoRARequest {
    pub fn new(
        lora_name: String,
        lora_int_id: u64,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Self {
        Self {
            lora_name,
            lora_int_id,
            lora_path,
            base_model_name: None,
            tensorizer_config_dict: None,
            load_inplace,
            is_3d_lora_weight,
        }
    }

    /// Convert this strongly typed request into the dynamic msgpack value used
    /// by `EngineCoreRequest.lora_request`.
    pub fn to_opaque_value(&self) -> Result<OpaqueValue> {
        rmpv::ext::to_value(self).map_err(|error| Error::Encode {
            target_type: type_name::<Self>(),
            message: format!("failed to encode LoRARequest: {}", error.to_report_string()),
        })
    }
}
