// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Controls how prompt rendering should end after the existing chat history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPromptMode {
    /// Append a generation prompt for a new assistant turn.
    ///
    /// Equivalent to `add_generation_prompt = true` and
    /// `continue_final_message = false`.
    #[default]
    StartNewAssistant,
    /// Leave the final assistant message open so generation continues it.
    ///
    /// Equivalent to `add_generation_prompt = false` and
    /// `continue_final_message = true`.
    ContinueFinalAssistant,
    /// Render the existing chat history without adding any trailing generation
    /// prompt.
    ///
    /// Equivalent to `add_generation_prompt = false` and
    /// `continue_final_message = false`.
    NoGenerationPrompt,
}

/// Effort level for reasoning models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable reasoning.
    None,
    /// Use the smallest available reasoning effort.
    Minimal,
    /// Use low reasoning effort.
    Low,
    /// Use medium reasoning effort.
    Medium,
    /// Use high reasoning effort.
    High,
    /// Use extra-high reasoning effort.
    XHigh,
    /// Use the largest available reasoning effort.
    Max,
}

impl ReasoningEffort {
    /// Return the lowercase value exposed to chat templates.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// Chat-template-related request options.
///
/// These are the chat controls that currently affect prompt rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatOptions {
    /// Controls whether rendering starts a new assistant turn, continues the
    /// final assistant message, or emits no trailing generation prompt.
    pub generation_prompt_mode: GenerationPromptMode,

    /// Per-request Jinja chat template override.
    ///
    /// The renderer uses this template in place of the model's default chat
    /// template when it is present.
    pub chat_template: Option<String>,

    /// Effort level exposed to chat templates for reasoning models.
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Additional keyword arguments exposed to the chat template.
    pub template_kwargs: HashMap<String, Value>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            chat_template: None,
            reasoning_effort: None,
            template_kwargs: HashMap::new(),
        }
    }
}

impl ChatOptions {
    /// Return whether rendering adds a prompt for a new assistant turn.
    pub fn add_generation_prompt(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::StartNewAssistant
        )
    }

    /// Return whether rendering continues the final assistant message.
    pub fn continue_final_message(&self) -> bool {
        matches!(
            self.generation_prompt_mode,
            GenerationPromptMode::ContinueFinalAssistant
        )
    }
}

/// Tool-choice semantics supported by the shared chat types.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolChoice {
    /// Disable tool calling.
    #[default]
    None,
    /// Let the model choose whether to call a tool.
    Auto,
    /// Require the model to call a tool.
    Required,
    /// Require one named function.
    Function {
        /// Required function name.
        name: String,
    },
}
