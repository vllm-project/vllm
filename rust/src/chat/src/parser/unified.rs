// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser registration and selection boundary for `vllm-chat`.

use std::sync::{Arc, LazyLock};

pub use vllm_parser::unified::{
    BeamUnifiedParser, DynamoUnifiedParser, Gemma4UnifiedParser, HfTemplateError, HfUnifiedParser,
    HyV3UnifiedParser, HyV4UnifiedParser, InklingUnifiedParser, KimiK3UnifiedParser,
    MuseUnifiedParser, REGISTERED_UNIFIED_FAMILIES, ResponseTemplate, UnifiedParser,
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
    pub const MUSE: &str = "muse";
    /// Alias of [`MUSE`]: the parser name of the original Muse Spark plugin.
    pub const MUSE_SPARK: &str = "musespark";

    /// Executes the checkpoint's Hugging Face `response_template`. Registered for
    /// name resolution only: it is built from the model's template, never by its
    /// registry constructor.
    pub const HF: &str = "hf";
    /// Beam checkpoints: [`HF`] with Beam's tool-call grammar, built from the
    /// model's template the same way.
    pub const BEAM: &str = "beam";
}

/// Constructor signature for one registered unified parser implementation.
#[derive(Clone)]
pub struct UnifiedParserCreator {
    create: Arc<UnifiedParserConstructor>,
    dynamo_family: Option<&'static str>,
}

type UnifiedParserConstructor = dyn Fn(&[ChatTool], DynTokenizer) -> vllm_parser::unified::Result<Box<dyn UnifiedParser>>
    + Send
    + Sync;

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
        Self::new_with_options(use_dynamo_rust_parser())
    }

    pub(crate) fn new_with_options(use_dynamo: bool) -> Self {
        let mut factory = Self::default();

        factory.register_parser::<Gemma4UnifiedParser>(names::GEMMA4);
        factory.register_parser::<HyV3UnifiedParser>(names::HY_V3);
        factory.register_parser::<HyV4UnifiedParser>(names::HY_V4);
        factory.register_parser::<InklingUnifiedParser>(names::INKLING);
        factory.register_parser::<KimiK3UnifiedParser>(names::KIMI_K3);
        factory.register_parser::<MuseUnifiedParser>(names::MUSE);
        factory.register_parser::<MuseUnifiedParser>(names::MUSE_SPARK);
        // Opt-in only: no model patterns map to it.
        factory.register_parser::<HfUnifiedParser>(names::HF);
        factory.register_parser::<BeamUnifiedParser>(names::BEAM);

        if use_dynamo {
            for &family in REGISTERED_UNIFIED_FAMILIES {
                factory.register_dynamo(family, family);
            }
            for (alias, family) in [
                ("qwen3_xml", "qwen3_coder"),
                (names::MUSE, "muse_spark"),
                (names::MUSE_SPARK, "muse_spark"),
            ] {
                factory.register_dynamo(alias, family);
            }
            factory
                .register_pattern("deepseek-v4.1", "deepseek_v41")
                .register_pattern("deepseek-v4", "deepseek_v4")
                .register_pattern("deepseek_v4", "deepseek_v4")
                .register_pattern("qwen3-coder", "qwen3_coder")
                .register_pattern("qwen3.5", "qwen3_coder")
                .register_pattern("qwen3-0.6b", "hermes")
                .register_pattern("hermes", "hermes")
                .register_pattern("mistral-4", "mistral_4")
                .register_pattern("nemotron-3", "nemotron_v3")
                .register_pattern("nemotron-v3", "nemotron_v3")
                .register_pattern("glm-5", "glm47")
                .register_pattern("glm-4.7", "glm47")
                .register_pattern("kimi-k2", "kimi_k2")
                .register_pattern("minimax-m3", "minimax_m3")
                .register_pattern("mm-m3", "minimax_m3")
                .register_pattern("muse-glimmer", "muse_glimmer")
                .register_pattern("muse-spark", "muse_spark")
                .register_pattern("gpt-oss", "gpt_oss")
                .register_pattern("gpt_oss", "gpt_oss")
                .register_pattern("beam", "beam");
        }

        factory
            .register_pattern("gemma-4", names::GEMMA4)
            .register_pattern("gemma4", names::GEMMA4)
            .register_pattern("hy3", names::HY_V3)
            .register_pattern("hy4", names::HY_V4)
            .register_pattern("inkling", names::INKLING)
            .register_pattern("kimi-k3", names::KIMI_K3)
            .register_pattern("kimi_k3", names::KIMI_K3)
            .register_pattern("muse-spark", names::MUSE);

        factory
    }

    /// Register one parser type that exposes a static `create()` constructor.
    pub fn register_parser<T>(&mut self, name: &str) -> &mut Self
    where
        T: UnifiedParser + 'static,
    {
        self.register_creator(
            name,
            UnifiedParserCreator {
                create: Arc::new(T::create),
                dynamo_family: None,
            },
        )
    }

    fn register_dynamo(&mut self, name: &str, family: &'static str) -> &mut Self {
        self.register_creator(
            name,
            UnifiedParserCreator {
                create: Arc::new(move |tools, tokenizer| {
                    Ok(Box::new(DynamoUnifiedParser::new(
                        family, tools, tokenizer,
                    )?))
                }),
                dynamo_family: Some(family),
            },
        )
    }

    pub fn dynamo_family(&self, name: &str) -> Option<&'static str> {
        self.creator(name).and_then(|creator| creator.dynamo_family)
    }

    pub fn equivalent_names(&self, left: Option<&str>, right: Option<&str>) -> bool {
        let canonical = |name: Option<&str>| {
            name.and_then(|name| self.dynamo_family(name))
                .and_then(vllm_parser::unified::canonical_unified_family)
        };
        left == right || (canonical(left).is_some() && canonical(left) == canonical(right))
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

        (creator.create)(tools, tokenizer).map_err(|error| crate::Error::ParserInitialization {
            kind: "unified",
            name: name.to_string(),
            error: error.into(),
        })
    }
}

pub(crate) fn use_dynamo_rust_parser() -> bool {
    std::env::var("VLLM_USE_DYNAMO_RUST_PARSER")
        .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "True" | "TRUE"))
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
            .with_regular_token("<|end_message|>", 200010)
            .with_regular_token("<|content_invoke_tool_json|>", 200049)
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
            .with_regular_token("<|sep|>", 1003)
            .with_regular_token("<|end_of_msg|>", 1004);

        assert!(factory.contains(names::KIMI_K3));
        assert_eq!(
            factory.resolve_name_for_model("moonshotai/Kimi-K3"),
            Some(names::KIMI_K3)
        );
        factory.create(names::KIMI_K3, &[], Arc::new(tokenizer)).unwrap();
    }

    #[test]
    fn factory_registers_muse_spark() {
        let factory = UnifiedParserFactory::new();
        let tokenizer = TestTokenizer::new()
            .with_regular_token("<|start|>", 200022)
            .with_regular_token("<|message|>", 200023)
            .with_regular_token("<|eom|>", 200007)
            .with_regular_token("<|eot|>", 200008);

        let model_id = "meta-models/Muse-Spark-open-816B-A42B-NVFP4";
        assert_eq!(factory.resolve_name_for_model(model_id), Some(names::MUSE));
        assert_eq!(
            crate::ParserSelection::Auto.resolve_reasoning_name(model_id),
            Some(names::MUSE)
        );
        assert_eq!(
            crate::ParserSelection::Auto.resolve_tool_name(model_id),
            Some(names::MUSE)
        );
        // Do not change automatic parser selection for existing Glimmer deployments.
        assert_eq!(
            factory.resolve_name_for_model("meta-models/Muse-Glimmer-30B"),
            None
        );
        for name in [names::MUSE, names::MUSE_SPARK] {
            factory.create(name, &[], Arc::new(tokenizer.clone())).unwrap();
            let explicit = crate::ParserSelection::Explicit(name.to_owned());
            crate::parser::validate_parser_overrides(&explicit, &explicit).unwrap();
        }
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
