use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for the Olmo3 family.
///
/// Olmo3 uses standard `<think>...</think>` delimiters but does not have them as
/// single tokens in its tokenizer vocabulary. It starts in reasoning mode by
/// default, and handles optionally stripping a response-leading `<think>`
/// prefix.
pub struct Olmo3ReasoningParser {
    inner: DelimitedReasoningParser,
    /// True until the first response text is classified. Only this position may
    /// drop a response-leading `<think>` prefix.
    at_response_start: bool,
    /// Holds an initial prefix like `<thi` while it may still complete into the
    /// leading opener on a later chunk.
    leading_start_buffer: String,
}

impl Olmo3ReasoningParser {
    /// Create an Olmo3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Self {
        Self {
            inner: DelimitedReasoningParser::new_optional(tokenizer, "<think>", "</think>", true),
            at_response_start: true,
            leading_start_buffer: String::new(),
        }
    }

    /// Drop a response-leading `<think>` prefix if present.
    fn push_inner(&mut self, delta: &str) -> ReasoningDelta {
        if self.at_response_start {
            self.leading_start_buffer.push_str(delta);
            let buffered = std::mem::take(&mut self.leading_start_buffer);

            if buffered.is_empty() {
                return ReasoningDelta::default();
            }

            if let Some(rest) = buffered.strip_prefix("<think>") {
                self.at_response_start = false;
                return self.inner.push(rest);
            }

            if "<think>".starts_with(buffered.as_str()) {
                self.leading_start_buffer = buffered;
                return ReasoningDelta::default();
            }

            self.at_response_start = false;
            return self.inner.push(&buffered);
        }

        self.inner.push(delta)
    }
}

fn append_delta(target: &mut ReasoningDelta, delta: ReasoningDelta) {
    if let Some(reasoning) = delta.reasoning {
        target.push_reasoning(&reasoning);
    }
    if let Some(content) = delta.content {
        target.push_content(&content);
    }
}

impl ReasoningParser for Olmo3ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        self.at_response_start = true;
        self.leading_start_buffer.clear();
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        Ok(self.push_inner(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let mut delta = ReasoningDelta::default();
        if !self.leading_start_buffer.is_empty() {
            let pending = std::mem::take(&mut self.leading_start_buffer);
            self.at_response_start = false;
            append_delta(&mut delta, self.inner.push(&pending));
        }
        append_delta(&mut delta, self.inner.finish());
        Ok(delta)
    }
}
