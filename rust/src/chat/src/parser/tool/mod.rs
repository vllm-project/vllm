// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tool parser registration and selection boundary for `vllm-chat`.

use std::sync::{Arc, LazyLock};

pub use vllm_parser::tool::{
    DeepSeekV3ToolParser, DeepSeekV4ToolParser, DeepSeekV31ToolParser, DeepSeekV32ToolParser,
    Glm45MoeToolParser, Glm47MoeToolParser, Granite4ToolParser, HermesToolParser, HyV3ToolParser,
    Internlm2ToolParser, KimiK2ToolParser, Llama3JsonToolParser, MinimaxM2ToolParser,
    MinimaxM3ToolParser, MistralToolParser, Phi4MiniJsonToolParser, Qwen3CoderToolParser,
    Qwen3XmlToolParser, SeedOssToolParser, ToolParser, ToolParserError,
};

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered tool parsers.
pub mod names {
    pub const DEEPSEEK_V3: &str = "deepseek_v3";
    pub const DEEPSEEK_V31: &str = "deepseek_v31";
    pub const DEEPSEEK_V32: &str = "deepseek_v32";
    pub const DEEPSEEK_V4: &str = "deepseek_v4";
    pub const GLM45: &str = "glm45";
    pub const GLM47: &str = "glm47";
    pub const GEMMA4: &str = "gemma4";
    pub const INKLING: &str = "inkling";
    pub const GRANITE4: &str = "granite4";
    pub const HERMES: &str = "hermes";
    pub const HY_V3: &str = "hy_v3";
    // Matches the Python CLI name `--tool-call-parser internlm`, which Python
    // also routes to `Internlm2ToolParser` despite the version-agnostic name.
    pub const INTERNLM: &str = "internlm";
    pub const KIMI_K2: &str = "kimi_k2";
    pub const LLAMA3_JSON: &str = "llama3_json";
    pub const LLAMA4_JSON: &str = "llama4_json";
    pub const MINIMAX_M2: &str = "minimax_m2";
    pub const MINIMAX_M3: &str = "minimax_m3";
    pub const MISTRAL: &str = "mistral";
    pub const PHI4_MINI_JSON: &str = "phi4_mini_json";
    pub const QWEN3_CODER: &str = "qwen3_coder";
    pub const QWEN3_XML: &str = "qwen3_xml";
    pub const SEED_OSS: &str = "seed_oss";
}

/// Constructor signature for one registered tool parser implementation.
type ToolParserCreator =
    Arc<dyn Fn(&[ChatTool]) -> vllm_parser::tool::Result<Box<dyn ToolParser>> + Send + Sync>;

/// Registry and model matcher for tool parsers.
pub type ToolParserFactory = ParserFactory<ToolParserCreator>;

impl ToolParserFactory {
    /// Get the global tool parser factory with built-in registrations and model
    /// mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ToolParserFactory> = LazyLock::new(ToolParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory
            .register_parser::<DeepSeekV3ToolParser>(names::DEEPSEEK_V3)
            .register_parser::<DeepSeekV31ToolParser>(names::DEEPSEEK_V31)
            .register_parser::<DeepSeekV32ToolParser>(names::DEEPSEEK_V32)
            .register_parser::<DeepSeekV4ToolParser>(names::DEEPSEEK_V4)
            .register_parser::<Glm45MoeToolParser>(names::GLM45)
            .register_parser::<Glm47MoeToolParser>(names::GLM47)
            .register_unified_dummy(names::GEMMA4)
            .register_unified_dummy(names::INKLING)
            .register_parser::<Granite4ToolParser>(names::GRANITE4)
            .register_parser::<HermesToolParser>(names::HERMES)
            .register_parser::<HyV3ToolParser>(names::HY_V3)
            .register_parser::<Internlm2ToolParser>(names::INTERNLM)
            .register_parser::<KimiK2ToolParser>(names::KIMI_K2)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA3_JSON)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA4_JSON)
            .register_parser::<MinimaxM2ToolParser>(names::MINIMAX_M2)
            .register_parser::<MinimaxM3ToolParser>(names::MINIMAX_M3)
            .register_parser::<MistralToolParser>(names::MISTRAL)
            .register_parser::<Phi4MiniJsonToolParser>(names::PHI4_MINI_JSON)
            .register_parser::<Qwen3XmlToolParser>(names::QWEN3_XML)
            .register_parser::<Qwen3CoderToolParser>(names::QWEN3_CODER)
            .register_parser::<SeedOssToolParser>(names::SEED_OSS);

        factory
            .register_pattern("mistral-", names::MISTRAL)
            .register_pattern("mixtral-", names::MISTRAL)
            .register_pattern("qwen3-coder", names::QWEN3_CODER)
            .register_pattern("qwen2.5-coder", names::QWEN3_CODER)
            .register_pattern("qwen3.5", names::QWEN3_CODER)
            .register_pattern("qwen", names::QWEN3_XML)
            .register_pattern("hermes", names::HERMES)
            .register_pattern("hy3", names::HY_V3)
            .register_pattern("hy_v3", names::HY_V3)
            // Narrow to `internlm2` substring so it matches `internlm2-chat-7b`
            // and `internlm2_5-7b-chat` but NOT `internlm-chat-7b` (InternLM v1,
            // routes to Llama), `internlm3-*` (also Llama-architecture per
            // vllm/model_executor/models/registry.py:146), or `Intern-S1` /
            // `Intern-S1-Pro` (separate intern-s1 parser, see PR #40115).
            .register_pattern("internlm2", names::INTERNLM)
            .register_pattern("llama-4", names::LLAMA4_JSON)
            .register_pattern("llama-3.2", names::LLAMA3_JSON)
            .register_pattern("llama-3.1", names::LLAMA3_JSON)
            .register_pattern("deepseek-r1", names::DEEPSEEK_V3)
            .register_pattern("deepseek-v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek_v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek-v3.2", names::DEEPSEEK_V32)
            .register_pattern("deepseek-v3.1", names::DEEPSEEK_V31)
            .register_pattern("deepseek-v3", names::DEEPSEEK_V3)
            .register_pattern("glm-5", names::GLM47)
            .register_pattern("glm-4.7", names::GLM47)
            .register_pattern("glm-4.6", names::GLM45)
            .register_pattern("glm-4.5", names::GLM45)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("granite-4", names::GRANITE4)
            .register_pattern("kimi-k2", names::KIMI_K2)
            .register_pattern("minimax-m3", names::MINIMAX_M3)
            .register_pattern("mm-m3", names::MINIMAX_M3)
            .register_pattern("minimax", names::MINIMAX_M2)
            .register_pattern("mm-m2", names::MINIMAX_M2)
            .register_pattern("seed-oss", names::SEED_OSS)
            .register_pattern("seedoss", names::SEED_OSS);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ToolParser + 'static,
    {
        self.register_creator(name, Arc::new(T::create))
    }

    /// Register one unified-only parser name in the split tool registry.
    pub fn register_unified_dummy(&mut self, name: &str) -> &mut Self {
        let name = name.to_string();
        let registered_name = name.clone();
        self.register_creator(
            &registered_name,
            Arc::new(move |_| Err(ToolParserError::DummyUnifiedParser { name: name.clone() })),
        )
    }

    /// Construct a parser from an exact name.
    pub fn create(&self, name: &str, tools: &[ChatTool]) -> crate::Result<Box<dyn ToolParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "tool",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator.as_ref()(tools).map_err(|error| crate::Error::ParserInitialization {
            kind: "tool",
            name: name.to_string(),
            error: error.into(),
        })
    }

    /// Resolve a parser from model ID and then construct it.
    pub fn create_for_model(
        &self,
        model_id: &str,
        tools: &[ChatTool],
    ) -> crate::Result<Box<dyn ToolParser>> {
        let name = self.resolve_name_for_model(model_id).ok_or_else(|| {
            crate::Error::ParserUnavailableForModel {
                kind: "tool",
                model_id: model_id.to_string(),
            }
        })?;
        self.create(name, tools)
    }
}

#[cfg(test)]
mod tests;
