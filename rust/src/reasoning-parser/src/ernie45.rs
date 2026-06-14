use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

const RESPONSE_START: &str = "<response>";
const RESPONSE_END: &str = "</response>";

/// Reasoning parser for Baidu ERNIE 4.5 thinking models.
///
/// The model output format is either of:
///
/// ```text
/// abc\n</think>\n\n<response>\ndef\n</response>\n
/// abc\n</think>\ndef
/// ```
///
/// Reasoning uses the standard `<think>...</think>` delimiters, with the model
/// typically starting directly inside a reasoning span and only emitting the
/// closing `</think>` (like DeepSeek R1), so the no-boundary fallback defaults
/// to `in_reasoning = true`.
///
/// On top of the shared delimited split, ERNIE 4.5 wraps the visible answer in
/// an optional `<response>...</response>` block. This parser strips those
/// structural markers from the content stream and trims the newline runs the
/// chat format inserts after `</think>` and `</response>`, so downstream
/// consumers see only the answer text. Text between the markers is forwarded
/// verbatim.
///
/// Unlike the Python parser, the `<response>` / `</response>` markers are
/// matched as plain text rather than by token ID, so they do not need to be
/// present in the tokenizer vocabulary. Also unlike Python — which truncates
/// everything after the final `</response>` — any non-newline text following
/// `</response>` is forwarded as content instead of being dropped.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/ernie45_reasoning_parser.py>
pub struct Ernie45ReasoningParser {
    inner: DelimitedReasoningParser,
    /// Holdback buffer for a partial `<response>` / `</response>` marker at
    /// the end of the content stream.
    content_buffer: String,
    /// Drop the structural newline run from upcoming content. Set after
    /// `</think>` and `</response>`, cleared at the first non-newline.
    trim_newlines: bool,
}

impl Ernie45ReasoningParser {
    /// Create an ERNIE 4.5 parser backed by the shared delimited state
    /// machine plus response-marker stripping.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", true)?,
            content_buffer: String::new(),
            trim_newlines: false,
        })
    }

    /// Post-process one inner delta: pass reasoning through and strip response
    /// markers plus structural newlines from the content half.
    fn postprocess(
        &mut self,
        inner_delta: ReasoningDelta,
        was_in_reasoning: bool,
    ) -> ReasoningDelta {
        let mut delta = ReasoningDelta::default();

        if let Some(reasoning) = &inner_delta.reasoning {
            delta.push_reasoning(reasoning);
        }

        // A reasoning -> content transition within this push means the content
        // half directly follows `</think>`: start trimming its newline run.
        // An explicit `<think>...</think>` round trip inside one push also
        // counts: the inner emits reasoning while ending in content mode.
        if !self.inner.in_reasoning() && (was_in_reasoning || inner_delta.reasoning.is_some()) {
            self.trim_newlines = true;
        }

        if let Some(content) = &inner_delta.content {
            let processed = self.process_content(content);
            delta.push_content(&processed);
        }

        delta
    }

    /// Buffer content text and process the prefix that cannot be part of a
    /// response marker split across deltas.
    fn process_content(&mut self, new_text: &str) -> String {
        self.content_buffer.push_str(new_text);

        let holdback = partial_marker_suffix_len(&self.content_buffer);
        let stable_len = self.content_buffer.len() - holdback;
        let pending_suffix = self.content_buffer.split_off(stable_len);
        let stable_text = std::mem::replace(&mut self.content_buffer, pending_suffix);

        self.process_stable_content(&stable_text)
    }

    /// Strip response markers and structural newline runs from content text
    /// that is known not to end with a partial marker suffix.
    fn process_stable_content(&mut self, mut stable: &str) -> String {
        let mut output = String::new();

        loop {
            if self.trim_newlines {
                stable = stable.trim_start_matches('\n');
                if stable.is_empty() {
                    break;
                }
                self.trim_newlines = false;
            }

            match find_earliest_marker(stable) {
                Some((marker_idx, marker)) => {
                    output.push_str(&stable[..marker_idx]);
                    stable = &stable[marker_idx + marker.len()..];
                    if marker == RESPONSE_END {
                        self.trim_newlines = true;
                    }
                }
                None => {
                    output.push_str(stable);
                    break;
                }
            }
        }

        output
    }
}

impl ReasoningParser for Ernie45ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        self.content_buffer.clear();
        self.trim_newlines = false;
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        let was_in_reasoning = self.inner.in_reasoning();
        let inner_delta = self.inner.push(delta);
        Ok(self.postprocess(inner_delta, was_in_reasoning))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let was_in_reasoning = self.inner.in_reasoning();
        let inner_delta = self.inner.finish();
        let mut delta = self.postprocess(inner_delta, was_in_reasoning);

        // A held-back partial marker prefix that never completed is real text.
        let leftover = std::mem::take(&mut self.content_buffer);
        if !leftover.is_empty() {
            delta.push_content(&leftover);
        }

        Ok(delta)
    }
}

/// Parse the longest text suffix that could still complete a response marker.
fn partial_marker_suffix_len(text: &str) -> usize {
    for (idx, _) in text.char_indices() {
        let suffix = &text[idx..];
        if (RESPONSE_START.starts_with(suffix) && suffix != RESPONSE_START)
            || (RESPONSE_END.starts_with(suffix) && suffix != RESPONSE_END)
        {
            return text.len() - idx;
        }
    }
    0
}

/// Parse the earliest complete response marker in the text, if any.
fn find_earliest_marker(text: &str) -> Option<(usize, &'static str)> {
    let start_idx = text.find(RESPONSE_START);
    let end_idx = text.find(RESPONSE_END);

    match (start_idx, end_idx) {
        (Some(start), Some(end)) if start < end => Some((start, RESPONSE_START)),
        (Some(_), Some(end)) => Some((end, RESPONSE_END)),
        (Some(start), None) => Some((start, RESPONSE_START)),
        (None, Some(end)) => Some((end, RESPONSE_END)),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Ernie45ReasoningParser;
    use crate::{ReasoningDelta, ReasoningParser, tests::FakeTokenizer};

    fn collect(parser: &mut Ernie45ReasoningParser, chunks: &[&str]) -> ReasoningDelta {
        let mut total = ReasoningDelta::default();
        for chunk in chunks {
            merge(&mut total, parser.push(chunk).unwrap());
        }
        merge(&mut total, parser.finish().unwrap());
        total
    }

    fn merge(total: &mut ReasoningDelta, delta: ReasoningDelta) {
        if let Some(reasoning) = &delta.reasoning {
            total.push_reasoning(reasoning);
        }
        if let Some(content) = &delta.content {
            total.push_content(content);
        }
    }

    #[test]
    fn ernie45_defaults_to_reasoning_without_prompt_boundary() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();

        let total = collect(&mut parser, &["abc\n</think>\ndef"]);

        assert_eq!(total.reasoning.as_deref(), Some("abc\n"));
        assert_eq!(total.content.as_deref(), Some("def"));
    }

    #[test]
    fn ernie45_strips_response_wrapper_and_structural_newlines() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();

        let total = collect(
            &mut parser,
            &["abc\n</think>\n\n<response>\ndef\n</response>\n"],
        );

        assert_eq!(total.reasoning.as_deref(), Some("abc\n"));
        assert_eq!(total.content.as_deref(), Some("\ndef\n"));
    }

    #[test]
    fn ernie45_streaming_handles_markers_split_across_deltas() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();

        let full = "abc\n</think>\n\n<response>\ndef\n</response>\n";
        let chunks: Vec<String> = full
            .as_bytes()
            .chunks(3)
            .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
            .collect();
        let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

        let total = collect(&mut parser, &chunk_refs);

        assert_eq!(total.reasoning.as_deref(), Some("abc\n"));
        assert_eq!(total.content.as_deref(), Some("\ndef\n"));
    }

    #[test]
    fn ernie45_flushes_partial_response_marker_on_finish() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();

        let total = collect(&mut parser, &["</think>x<resp"]);

        assert_eq!(total.reasoning, None);
        assert_eq!(total.content.as_deref(), Some("x<resp"));
    }

    #[test]
    fn ernie45_explicit_think_round_trip_still_trims_after_think_end() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[2]).unwrap();

        let total = collect(&mut parser, &["<think>more\n</think>\nanswer"]);

        assert_eq!(total.reasoning.as_deref(), Some("more\n"));
        assert_eq!(total.content.as_deref(), Some("answer"));
    }

    #[test]
    fn ernie45_prompt_end_marker_starts_in_content_without_trimming() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[2]).unwrap();

        let total = collect(&mut parser, &["hello"]);

        assert_eq!(total.reasoning, None);
        assert_eq!(total.content.as_deref(), Some("hello"));
    }

    #[test]
    fn ernie45_forwards_text_after_response_end() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();

        let total = collect(
            &mut parser,
            &["abc</think><response>def</response>\ntrailing"],
        );

        assert_eq!(total.reasoning.as_deref(), Some("abc"));
        assert_eq!(total.content.as_deref(), Some("deftrailing"));
    }

    #[test]
    fn ernie45_keeps_response_markers_inside_reasoning() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = Ernie45ReasoningParser::new(tokenizer).unwrap();

        let total = collect(&mut parser, &["plan: emit <response> later</think>ok"]);

        assert_eq!(
            total.reasoning.as_deref(),
            Some("plan: emit <response> later")
        );
        assert_eq!(total.content.as_deref(), Some("ok"));
    }
}
