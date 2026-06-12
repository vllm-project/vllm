//! Reasoning parser registration and selection boundary for `vllm-chat`.

use std::sync::LazyLock;

pub use vllm_reasoning_parser::{
    CohereCmdReasoningParser, DeepSeekR1ReasoningParser, DeepSeekV3ReasoningParser,
    DeepSeekV4ReasoningParser, Gemma4ReasoningParser, Glm45ReasoningParser, KimiK2ReasoningParser,
    KimiReasoningParser, MiniMaxM2ReasoningParser, NemotronV3ReasoningParser, Qwen3ReasoningParser,
    ReasoningDelta, ReasoningError, ReasoningParser, Step3ReasoningParser,
};
use vllm_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;

/// Canonical public names for registered reasoning parsers.
pub mod names {
    pub const COHERE_CMD: &str = "cohere_cmd";
    pub const DEEPSEEK_R1: &str = "deepseek_r1";
    pub const DEEPSEEK_V3: &str = "deepseek_v3";
    pub const DEEPSEEK_V4: &str = "deepseek_v4";
    pub const GEMMA4: &str = "gemma4";
    pub const GLM45: &str = "glm45";
    pub const KIMI: &str = "kimi";
    pub const KIMI_K2: &str = "kimi_k2";
    pub const MINIMAX_M2: &str = "minimax_m2";
    pub const NEMOTRON_V3: &str = "nemotron_v3";
    pub const QWEN3: &str = "qwen3";
    pub const STEP3: &str = "step3";
}

/// Constructor signature for one registered reasoning parser implementation.
type ReasoningParserCreator =
    fn(DynTokenizer) -> vllm_reasoning_parser::Result<Box<dyn ReasoningParser>>;

/// Registry and model matcher for reasoning parsers.
pub type ReasoningParserFactory = ParserFactory<ReasoningParserCreator>;

impl ReasoningParserFactory {
    /// Get the global reasoning parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<ReasoningParserFactory> =
            LazyLock::new(ReasoningParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory
            .register_parser::<CohereCmdReasoningParser>(names::COHERE_CMD)
            .register_parser::<DeepSeekR1ReasoningParser>(names::DEEPSEEK_R1)
            .register_parser::<DeepSeekV3ReasoningParser>(names::DEEPSEEK_V3)
            .register_parser::<DeepSeekV4ReasoningParser>(names::DEEPSEEK_V4)
            .register_parser::<Gemma4ReasoningParser>(names::GEMMA4)
            .register_parser::<Glm45ReasoningParser>(names::GLM45)
            .register_parser::<KimiReasoningParser>(names::KIMI)
            .register_parser::<KimiK2ReasoningParser>(names::KIMI_K2)
            .register_parser::<MiniMaxM2ReasoningParser>(names::MINIMAX_M2)
            .register_parser::<NemotronV3ReasoningParser>(names::NEMOTRON_V3)
            .register_parser::<Qwen3ReasoningParser>(names::QWEN3)
            .register_parser::<Step3ReasoningParser>(names::STEP3);

        factory
            .register_pattern("deepseek-r1", names::DEEPSEEK_R1)
            .register_pattern("deepseek-v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek_v4", names::DEEPSEEK_V4)
            .register_pattern("deepseek-v3", names::DEEPSEEK_V3)
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("qwen", names::QWEN3)
            .register_pattern("glm-5", names::GLM45)
            .register_pattern("glm-4.7", names::GLM45)
            .register_pattern("glm-4.6", names::GLM45)
            .register_pattern("glm-4.5", names::GLM45)
            .register_pattern("kimi-k2", names::KIMI_K2)
            .register_pattern("kimi", names::KIMI)
            .register_pattern("step3", names::STEP3)
            .register_pattern("minimax", names::MINIMAX_M2)
            .register_pattern("mm-m2", names::MINIMAX_M2)
            .register_pattern("cohere", names::COHERE_CMD)
            .register_pattern("command", names::COHERE_CMD)
            .register_pattern("nano", names::NEMOTRON_V3)
            .register_pattern("nemotron", names::NEMOTRON_V3);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: ReasoningParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn ReasoningParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "reasoning",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "reasoning",
            name: name.to_string(),
            error: error.into(),
        })
    }
}

#[cfg(test)]
mod tests;
