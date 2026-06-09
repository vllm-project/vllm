use vllm_tokenizer::DynTokenizer;

use super::{
    DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result,
};

/// Reasoning parser for Qwen3 style outputs.
///
/// Qwen3 uses the standard `<think>...</think>` delimiters, but its
/// initialization must respect special tokens that may appear after a start
/// delimiter in the prompt. If a special token is encountered after the start
/// token, the parser should stop scanning for reasoning and treat all further
/// output as plain content.
pub struct Qwen3ReasoningParser {
    inner: DelimitedReasoningParser,
    tokenizer: DynTokenizer,
    /// When `true`, the parser will split reasoning/content using the inner
    /// delimited parser. When `false`, all output is treated as content.
    allow_reasoning: bool,
}

impl Qwen3ReasoningParser {
    /// Create a new Qwen3 parser backed by a shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let inner = DelimitedReasoningParser::new(
            tokenizer.clone(),
            "<think>",
            "</think>",
            false,
        )?;
        Ok(Self {
            inner,
            tokenizer,
            // Default to allowing reasoning; actual value will be set in `initialize`.
            allow_reasoning: true,
        })
    }
}

impl ReasoningParser for Qwen3ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        // Determine whether reasoning should be parsed. We start in reasoning mode
        // if a `<think>` token appears in the prompt and no special token follows
        // it. If a special token appears after the start token, we disable reasoning.
        let start_token_id = self.tokenizer.token_to_id("<think>");
        let mut seen_start = false;
        let mut allow = true;

        for &tid in prompt_token_ids {
            if let Some(start_id) = start_token_id {
                if tid == start_id {
                    seen_start = true;
                }
            }
            if seen_start && self.tokenizer.is_special_id(tid) {
                // A special token after the start delimiter – stop reasoning.
                allow = false;
                break;
            }
        }

        self.allow_reasoning = allow;
        self.inner.initialize(prompt_token_ids)
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        if self.allow_reasoning {
            self.inner.push(delta)
        } else {
            Ok(ReasoningDelta {
                reasoning: None,
                content: Some(delta.to_string()),
            })
        }
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        if self.allow_reasoning {
            self.inner.finish()
        } else {
            Ok(ReasoningDelta::default())
        }
    }
}
