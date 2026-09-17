// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::protocol::OpaqueValue;
use crate::protocol::tensor::WireNdArray;

/// Python wire representation of `LogprobsLists` / `LogprobsTensors` before
/// aux-frame references and raw-view payloads are resolved.
///
/// This mirrors the tuple shape emitted by Python engine-core so serde can
/// first deserialize the raw wire payload before the Rust client converts it
/// into semantic per-position logprobs records.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/d5cadcee8641a9fcec15facb5a9157d157daa207/vllm/v1/outputs.py#L30-L106>
///
/// # Wire compatibility
///
/// Python's `LogprobsLists` / `LogprobsTensors` are `NamedTuple`s: every
/// field is serialized positionally, and `Deserialize_tuple` here hard-fails
/// on payloads longer than this struct. To preserve positional compatibility,
/// append new Python fields and add matching trailing `#[serde(default)]`
/// fields here in the same change.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireLogprobs {
    /// Wire array with shape `[num_positions, max_num_logprobs + 1]`.
    pub logprob_token_ids: WireNdArray,
    /// Wire array with shape `[num_positions, max_num_logprobs + 1]`.
    pub logprobs: WireNdArray,
    /// Wire array with shape `[num_positions]`.
    ///
    /// Python uses the field name `sampled_token_ranks` for sample logprobs and
    /// `selected_token_ranks` for prompt logprobs. Rust keeps one neutral field
    /// because both payloads share the same wire representation.
    pub token_ranks: WireNdArray,
    /// Preserved only for wire compatibility with batch-level Python tensors.
    /// Scheduler-sliced per-request outputs should emit `None` here, and
    /// the semantic Rust decoder rejects any other value.
    #[serde(default)]
    pub cu_num_generated_tokens: Option<Vec<usize>>,
    /// Device-only generation boundaries added to `LogprobsTensors` in
    /// vllm-project/vllm#52242. Preserved only for wire compatibility:
    /// NamedTuple encoding always emits the slot even when it is `None`,
    /// so this field must exist for the 5-element payload to decode.
    /// Scheduler-sliced per-request outputs should emit `None` here, and
    /// the semantic Rust decoder rejects any other value.
    #[serde(default)]
    pub cu_num_generated_tokens_tensor: Option<OpaqueValue>,
}
