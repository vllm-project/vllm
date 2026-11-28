// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Map, Value};
use vllm_text::backend::hf::{HfOverrides, HfToken, ModelSource};
use vllm_text::{DynTextBackend, GenerationConfigMode, TextBackend};

use crate::error::Result;
use crate::multimodal::{MmLimitPerPrompt, MultimodalModelInfo};
use crate::output::DynChatOutputProcessor;
use crate::renderer::DynChatRenderer;
use crate::request::ChatRequest;
use crate::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection, ToolStrictLevel};

pub mod hf;
mod tokenizerless;

/// Options for creating a new chat output processor.
pub struct NewChatOutputProcessorOptions {
    /// Server-side floor for tool-call structural tags, or `None` for the
    /// tool parser's default level.
    pub tool_strict_level: Option<ToolStrictLevel>,
}

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: Send + Sync {
    /// Return the renderer used for chat-prompt construction.
    fn chat_renderer(&self) -> DynChatRenderer;

    /// Return model files/config needed for request-scoped multimodal
    /// preprocessing, if supported.
    fn multimodal_model_info(&self) -> Option<&MultimodalModelInfo> {
        None
    }

    /// Create a request-scoped output processor after request-level adjustments
    /// are applied.
    fn new_chat_output_processor(
        &self,
        request: &mut ChatRequest,
        options: NewChatOutputProcessorOptions,
    ) -> Result<DynChatOutputProcessor>;

    /// Effective tool-call parser name, if the output processors parse tool
    /// calls with a selectable parser.
    fn tool_call_parser_name(&self) -> Option<&str> {
        None
    }

    /// Effective reasoning parser name, if the output processors parse
    /// reasoning with a selectable parser.
    fn reasoning_parser_name(&self) -> Option<&str> {
        None
    }
}

/// Shared trait-object form of [`ChatBackend`].
pub type DynChatBackend = Arc<dyn ChatBackend>;

/// Convenience trait for backends that can serve both raw text generation and
/// chat templating.
///
/// This is mainly useful in tests and small examples, where one mock/backend
/// often implements both sides and callers want `ChatLlm` to wire the shared
/// object into `TextLlm` automatically.
pub trait ChatTextBackend: ChatBackend + TextBackend {}

impl<T> ChatTextBackend for T where T: ChatBackend + TextBackend + ?Sized {}

/// Shared trait-object form of [`ChatTextBackend`].
pub type DynChatTextBackend = Arc<dyn ChatTextBackend>;

/// Frontend-side chat backend loading options.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoadModelBackendsOptions {
    /// Model revision on the Hugging Face Hub (branch, tag, or commit SHA).
    pub revision: Option<String>,
    /// Source of the tokenizer files and the chat template, or `None` for the model.
    pub tokenizer: Option<ModelSource>,
    /// Source of `config.json` and `generation_config.json`, or `None` for the model.
    pub hf_config: Option<ModelSource>,
    /// Load no tokenizer and serve token IDs only, rejecting chat requests and
    /// text prompts (`--skip-tokenizer-init`).
    pub skip_tokenizer_init: bool,
    /// Token for Hub downloads (`--hf-token`), or `None` for `HF_TOKEN` and
    /// the stored login token.
    pub hf_token: Option<HfToken>,
    /// JSON Merge Patch applied to the model config before loading any backend.
    pub hf_overrides: HfOverrides,
    /// Which generation-config sampling defaults to inherit.
    pub generation_config: GenerationConfigMode,
    /// Keys applied over the generation-config sampling defaults.
    pub override_generation_config: Map<String, Value>,
    /// Which chat renderer implementation to use.
    pub renderer: RendererSelection,
    /// Tool-call parser selection, resolved for the model at load.
    pub tool_call_parser: ParserSelection,
    /// Reasoning parser selection, resolved for the model at load.
    pub reasoning_parser: ParserSelection,
    /// Disable frontend-side multimodal preprocessing and render the model as
    /// language-only.
    pub language_model_only: bool,
    /// How to serialize `message.content` when rendering the chat template.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Optional server-default chat template override, provided either as an
    /// inline template or as a path to a template file.
    pub chat_template: Option<String>,
    /// Whether requests may override the chat template.
    pub trust_request_chat_template: bool,
    /// Server-default keyword arguments. HF merges these before request kwargs;
    /// native renderers inherit supported reasoning controls below request controls.
    pub default_chat_template_kwargs: HashMap<String, Value>,
    /// Maximum number of input items allowed per prompt for each modality.
    /// Unspecified modalities default to the model's default, else 999 items.
    pub limit_mm_per_prompt: MmLimitPerPrompt,
    /// Default overrides for multimodal processor options.
    pub mm_processor_kwargs: HashMap<String, Value>,
}

/// Shared backends loaded from a model id.
pub struct LoadedModelBackends {
    pub text_backend: DynTextBackend,
    pub chat_backend: DynChatBackend,
    /// The concrete renderer selected after model-based auto-detection.
    pub renderer: Option<RendererSelection>,
    /// The tokenizer implementation that produced prompt token IDs.
    pub tokenizer_library: Option<String>,
    /// Resolved files shared by the text and chat backends.
    pub resolved_model_files: vllm_text::backend::hf::ResolvedModelFiles,
}

/// Load text and chat backends for the given model id.
pub async fn load_model_backends(
    model_id: &str,
    options: LoadModelBackendsOptions,
) -> Result<LoadedModelBackends> {
    // Currently, we only have HuggingFace backends.
    hf::load_model_backends(model_id, options).await
}
