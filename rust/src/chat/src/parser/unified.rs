// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser registration and selection boundary for `vllm-chat`.

use std::sync::LazyLock;

pub use vllm_parser::unified::{Gemma4UnifiedParser, InklingUnifiedParser, UnifiedParser};
use vllm_tokenizer::DynTokenizer;

use crate::parser::ParserFactory;
use crate::request::ChatTool;

/// Canonical public names for registered unified parsers.
pub mod names {
    pub const GEMMA4: &str = "gemma4";
    pub const INKLING: &str = "inkling";
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
        factory.register_parser::<InklingUnifiedParser>(names::INKLING);

        factory
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("inkling", names::INKLING);

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
}
