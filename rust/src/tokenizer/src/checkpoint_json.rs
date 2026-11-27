// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::de::DeserializeOwned;

pub use json5::Error as CheckpointJsonError;

/// Parse a JSON file shipped with a model checkpoint, such as `config.json`,
/// `tokenizer_config.json`, a processor config, or `chat_template.json`.
///
/// Python reads these files with `json`, which accepts bare `NaN`, `Infinity`,
/// and `-Infinity` (some Mamba configs set `"time_step_limit": [0.0, Infinity]`),
/// so they are parsed with `json5` instead of `serde_json`. Read into a
/// `serde_json::Value`, such a number becomes `null`. JSON that is not a
/// checkpoint file, such as request bodies and model output, stays on
/// `serde_json`.
pub fn parse_checkpoint_json<T: DeserializeOwned>(text: &str) -> Result<T, CheckpointJsonError> {
    json5::from_str(text)
}
