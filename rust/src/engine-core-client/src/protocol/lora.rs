// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

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
    pub load_inplace: bool,
    #[serde(default)]
    pub is_3d_lora_weight: bool,
}

/// One measured adapter transition on the worker.
///
/// Mirrors Python `vllm.lora.load_event.LoRALoadTiming`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraLoadTiming {
    pub adapter_name: String,
    /// `load`: read from disk into the CPU cache. `activate`: moved from the
    /// CPU cache into a GPU slot.
    pub transition: String,
    pub seconds: f64,
}

/// Full snapshot of a worker's adapter caches, keyed by adapter name.
///
/// Mirrors Python `vllm.lora.load_event.LoRALoadEvent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoraLoadEvent {
    /// Adapters activated into GPU slots (sorted).
    #[serde(default)]
    pub gpu_adapters: Vec<String>,
    /// Adapters resident in the CPU cache (sorted, superset of `gpu_adapters`).
    #[serde(default)]
    pub cpu_adapters: Vec<String>,
    /// Adapters pinned in the caches (sorted).
    #[serde(default)]
    pub pinned_adapters: Vec<String>,
    /// LoRA rank of each resident adapter, keyed by name.
    #[serde(default)]
    pub ranks: BTreeMap<String, u32>,
    /// Adapter transitions completed since the previous event, in order.
    #[serde(default)]
    pub loads: Vec<LoraLoadTiming>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::protocol::decode_msgpack;
    use crate::protocol::lora::{LoraLoadEvent, LoraLoadTiming};

    /// Python: `encode(LoRALoadEvent(gpu_adapters=["alpha"],
    /// cpu_adapters=["alpha", "beta"], pinned_adapters=["alpha"],
    /// ranks={"alpha": 64, "beta": 8}, loads=[LoRALoadTiming("alpha", "load",
    /// 0.25), LoRALoadTiming("alpha", "activate", 0.5)]))`
    const PYTHON_LORA_LOAD_EVENT: &str = "85ac6770755f616461707465727391a5616c706861ac6370755f616461707465727392a5616c706861a462657461af70696e6e65645f616461707465727391a5616c706861a572616e6b7382a5616c70686140a46265746108a56c6f6164739283ac616461707465725f6e616d65a5616c706861aa7472616e736974696f6ea46c6f6164a77365636f6e6473cb3fd000000000000083ac616461707465725f6e616d65a5616c706861aa7472616e736974696f6ea86163746976617465a77365636f6e6473cb3fe0000000000000";

    #[test]
    fn lora_load_event_decodes_python_encoding() {
        let event: LoraLoadEvent =
            decode_msgpack(&hex::decode(PYTHON_LORA_LOAD_EVENT).unwrap()).unwrap();
        assert_eq!(
            event,
            LoraLoadEvent {
                gpu_adapters: vec!["alpha".to_string()],
                cpu_adapters: vec!["alpha".to_string(), "beta".to_string()],
                pinned_adapters: vec!["alpha".to_string()],
                ranks: BTreeMap::from([("alpha".to_string(), 64), ("beta".to_string(), 8)]),
                loads: vec![
                    LoraLoadTiming {
                        adapter_name: "alpha".to_string(),
                        transition: "load".to_string(),
                        seconds: 0.25,
                    },
                    LoraLoadTiming {
                        adapter_name: "alpha".to_string(),
                        transition: "activate".to_string(),
                        seconds: 0.5,
                    },
                ],
            }
        );
    }

    /// Python: `encode(LoRALoadEvent())`.
    #[test]
    fn lora_load_event_decodes_omitted_defaults() {
        let event: LoraLoadEvent = decode_msgpack(&hex::decode("80").unwrap()).unwrap();
        assert_eq!(event, LoraLoadEvent::default());
    }
}
