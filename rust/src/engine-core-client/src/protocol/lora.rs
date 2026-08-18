// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_tuple::{Deserialize_tuple, Serialize_tuple};
use thiserror::Error;

use crate::protocol::OpaqueValue;

/// Request for a LoRA adapter.
///
/// Mirrors Python `vllm.lora.request.LoRARequest`, which is a msgspec
/// `array_like=True` struct. Keep the field order aligned with Python.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct LoraRequest {
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

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LoraRequestError {
    #[error("lora_name must not be empty")]
    EmptyName,
    #[error("lora_int_id must be nonzero")]
    ZeroId,
    #[error("lora_path must not be empty")]
    EmptyPath,
}

impl LoraRequest {
    pub fn new(
        lora_name: String,
        lora_int_id: u64,
        lora_path: String,
        load_inplace: bool,
        is_3d_lora_weight: bool,
    ) -> Result<Self, LoraRequestError> {
        if lora_name.trim().is_empty() {
            return Err(LoraRequestError::EmptyName);
        }
        if lora_int_id == 0 {
            return Err(LoraRequestError::ZeroId);
        }
        if lora_path.trim().is_empty() {
            return Err(LoraRequestError::EmptyPath);
        }

        Ok(Self {
            lora_name,
            lora_int_id,
            lora_path,
            base_model_name: None,
            tensorizer_config_dict: None,
            load_inplace,
            is_3d_lora_weight,
        })
    }
}
