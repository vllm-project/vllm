//! Tool parser registration and selection boundary for `vllm-chat`.

use std::sync::LazyLock;

pub use vllm_tool_parser::{
    DeepSeekV3ToolParser, DeepSeekV4ToolParser, DeepSeekV31ToolParser, DeepSeekV32ToolParser,
    Gemma4ToolParser, Glm45MoeToolParser, Glm47MoeToolParser, HermesToolParser, KimiK2ToolParser,
    Llama3JsonToolParser, MinimaxM2ToolParser, MistralToolParser, Qwen3CoderToolParser,
    Qwen3XmlToolParser, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError,
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
    pub const HERMES: &str = "hermes";
    pub const KIMI_K2: &str = "kimi_k2";
    pub const LLAMA3_JSON: &str = "llama3_json";
    pub const LLAMA4_JSON: &str = "llama4_json";
    pub const MINIMAX_M2: &str = "minimax_m2";
    pub const MISTRAL: &str = "mistral";
    pub const QWEN3_CODER: &str = "qwen3_coder";
    pub const QWEN3_XML: &str = "qwen3_xml";
}

/// Constructor signature for one registered tool parser implementation.
type ToolParserCreator = fn(&[ChatTool]) -> vllm_tool_parser::Result<Box<dyn ToolParser>>;

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
            .register_parser::<Gemma4ToolParser>(names::GEMMA4)
            .register_parser::<HermesToolParser>(names::HERMES)
            .register_parser::<KimiK2ToolParser>(names::KIMI_K2)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA3_JSON)
            .register_parser::<Llama3JsonToolParser>(names::LLAMA4_JSON)
            .register_parser::<MinimaxM2ToolParser>(names::MINIMAX_M2)
            .register_parser::<MistralToolParser>(names::MISTRAL)
            .register_parser::<Qwen3XmlToolParser>(names::QWEN3_XML)
            .register_parser::<Qwen3CoderToolParser>(names::QWEN3_CODER);

        factory
            .register_pattern("mistral-", names::MISTRAL)
            .register_pattern("mixtral-", names::MISTRAL)
            .register_pattern("qwen3-coder", names::QWEN3_CODER)
            .register_pattern("qwen2.5-coder", names::QWEN3_CODER)
            .register_pattern("qwen3.5", names::QWEN3_CODER)
            .register_pattern("qwen", names::QWEN3_XML)
            .register_pattern("hermes", names::HERMES)
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
            .register_pattern("kimi-k2", names::KIMI_K2)
            .register_pattern("minimax", names::MINIMAX_M2);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ToolParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(&self, name: &str, tools: &[ChatTool]) -> crate::Result<Box<dyn ToolParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "tool",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tools).map_err(|error| crate::Error::ParserInitialization {
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
