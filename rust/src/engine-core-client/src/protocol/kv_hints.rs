// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvHintAction {
    pub action_id: String,
    pub action_type: String,
    pub action_version: String,
    pub payload: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvHintsEnvelope {
    pub protocol_version: String,
    pub message_id: String,
    pub actions: Vec<KvHintAction>,
}
