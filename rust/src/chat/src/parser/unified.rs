// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser registration and selection boundary for `vllm-chat`.

use std::sync::LazyLock;

pub use vllm_parser::unified::{
    Gemma4UnifiedParser, HyV3UnifiedParser, HyV4UnifiedParser, InklingUnifiedParser,
    KimiK3UnifiedParser, UnifiedParser,
};
use vllm_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered unified parsers.
pub mod names {
    pub const GEMMA4: &str = "gemma4";
    pub const HY_V3: &str = "hy_v3";
    pub const HY_V4: &str = "hy_v4";
    pub const INKLING: &str = "inkling";
    pub const KIMI_K3: &str = "kimi_k3";
}

/// Constructor signature for one registered unified parser implementation.
type UnifiedParserCreator =
    fn(&[ChatTool], DynTokenizer) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>>;

/// Registry and model matcher for unified parsers.
pub type UnifiedParserFactory = ParserFactory<UnifiedParserCreator>;

impl UnifiedParserFactory {
    /// Get the global unified parser factory with built-in registrations and
    /// model mappings.
    pub fn global() -> &'static Self {
        static INSTANCE: LazyLock<UnifiedParserFactory> = LazyLock::new(UnifiedParserFactory::new);
        &INSTANCE
    }

    /// Create the default registry with built-in parser names and model
    /// mappings.
    pub fn new() -> Self {
        let mut factory = Self::default();

        factory.register_parser::<Gemma4UnifiedParser>(names::GEMMA4);
        factory.register_parser::<HyV3UnifiedParser>(names::HY_V3);
        factory.register_parser::<HyV4UnifiedParser>(names::HY_V4);
        factory.register_parser::<InklingUnifiedParser>(names::INKLING);
        factory.register_parser::<KimiK3UnifiedParser>(names::KIMI_K3);

        factory
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("hy3", names::HY_V3)
            .register_pattern("hy4", names::HY_V4)
            .register_pattern("inkling", names::INKLING)
            .register_pattern("kimi-k3", names::KIMI_K3)
            .register_pattern("kimi_k3", names::KIMI_K3);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: UnifiedParser + 'static,
    {
        self.register_creator(name, T::create)
    }

    /// Construct a parser from an exact name.
    pub fn create(
        &self,
        name: &str,
        tools: &[ChatTool],
        tokenizer: DynTokenizer,
    ) -> crate::Result<Box<dyn UnifiedParser>> {
        let creator = self.creator(name).ok_or_else(|| crate::Error::ParserUnavailableByName {
            kind: "unified",
            name: name.to_string(),
            available_names: self.list(),
        })?;

        creator(tools, tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "unified",
            name: name.to_string(),
            error: error.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{UnifiedParserFactory, names};

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|channel>", 256)
            .with_regular_token("<channel|>", 257)
    }

    fn inkling_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_regular_token("<|message_model|>", 200001)
            .with_regular_token("<|content_text|>", 200004)
            .with_regular_token("<|content_thinking|>", 200008)
    }

    #[test]
    fn factory_registers_gemma4() {
        let factory = UnifiedParserFactory::new();

        assert!(factory.contains(names::GEMMA4));
        assert_eq!(
            factory.resolve_name_for_model("google/gemma-4-27b-it"),
            Some(names::GEMMA4)
        );
        factory.create(names::GEMMA4, &[], Arc::new(tokenizer())).unwrap();
    }

    #[test]
    fn factory_registers_inkling() {
        let factory = UnifiedParserFactory::new();

        assert!(factory.contains(names::INKLING));
        assert_eq!(
            factory.resolve_name_for_model("thinkingmachines/Inkling"),
            Some(names::INKLING)
        );
        factory.create(names::INKLING, &[], Arc::new(inkling_tokenizer())).unwrap();
    }

    #[test]
    fn factory_registers_kimi_k3() {
        let factory = UnifiedParserFactory::new();
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|open|>", 1001)
            .with_regular_token("<|close|>", 1002)
            .with_regular_token("<|sep|>", 1003);

        assert!(factory.contains(names::KIMI_K3));
        assert_eq!(
            factory.resolve_name_for_model("moonshotai/Kimi-K3"),
            Some(names::KIMI_K3)
        );
        factory.create(names::KIMI_K3, &[], Arc::new(tokenizer)).unwrap();
    }

    #[test]
    fn factory_registers_hy_v4() {
        let factory = UnifiedParserFactory::new();
        let tokenizer = [
            "<think:opensource>",
            "</think:opensource>",
            "<tool_calls:opensource>",
            "</tool_calls:opensource>",
            "<tool_call:opensource>",
            "</tool_call:opensource>",
            "<arg_key:opensource>",
            "</arg_key:opensource>",
            "<arg_value:opensource>",
            "</arg_value:opensource>",
        ]
        .into_iter()
        .enumerate()
        .fold(TestTokenizer::new(), |tokenizer, (index, token)| {
            tokenizer.with_regular_token(token, 1000 + index as u32)
        });

        assert!(factory.contains(names::HY_V4));
        assert_eq!(
            factory.resolve_name_for_model("tencent/Hy4-preview"),
            Some(names::HY_V4)
        );
        factory.create(names::HY_V4, &[], Arc::new(tokenizer)).unwrap();
    }
}
