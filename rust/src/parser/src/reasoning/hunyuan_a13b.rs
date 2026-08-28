// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Opening delimiter of the reasoning section.
const THINK_START: &str = "<think>";
/// Reasoning ends where the answer section begins.
///
/// The leading newline belongs to the delimiter: Python matches
/// `(.*?)\n</think>\n<answer>\n`, so the newline before `</think>` is framing
/// rather than reasoning. The trailing newline is deliberately left out, see
/// [`HunyuanA13BReasoningParser`].
const ANSWER_START: &str = "\n</think>\n<answer>";
/// Opening marker of the answer section, seen alone only when the prompt
/// prefilled and closed the reasoning section.
const ANSWER_OPEN: &str = "<answer>";
/// Marker closing the answer section. Protocol framing, not content.
const ANSWER_END: &str = "\n</answer>";

/// Reasoning parser for Hunyuan A13B outputs.
///
/// Hunyuan A13B frames its output as `<think>\n` reasoning
/// `\n</think>\n<answer>\n` content `\n</answer>`, making it the only family
/// here that wraps its content in a section of its own. The shared delimited
/// state machine does the reasoning split and drops the framing newline that
/// opens each section; this wrapper suppresses the trailing `\n</answer>` and
/// the bare `<answer>` of a prefilled prompt.
///
/// None of these delimiters is a vocabulary token: `<think>\n` tokenizes as
/// `<th` + `ink` + `>Ċ`, which is why the Python parser hardcodes literal ID
/// sequences such as `[14023, 771, 397]`. Matching text keeps the parser working
/// across tokenizer revisions, at the cost of having no prompt token boundary to
/// initialize from. That costs nothing here: with thinking enabled the chat
/// template prefills no reasoning at all and the model emits `<think>\n` itself.
///
/// The delimiters end just before their framing newline on purpose. The shared
/// machine holds back any buffered suffix that could still complete *either*
/// delimiter, regardless of which one it is currently looking for, so a pair
/// where one delimiter ends with the other's opening character cannot be
/// matched: after `<think>\n` the trailing newline would be held back as a
/// possible start of `\n</think>`, leaving `<think>` to be emitted as content.
/// Ending both delimiters on `>` avoids the overlap, and costs only the single
/// leading newline, which the shared machine strips per section via
/// [`DelimitedReasoningParser::strip_framing_newlines`].
// TODO: the Python parser also exposes `is_reasoning_end` and
// `extract_content_ids` over token IDs. The Rust `ReasoningParser` trait has no
// equivalent hook yet, so neither is ported here.
pub struct HunyuanA13BReasoningParser {
    inner: DelimitedReasoningParser,
    /// A bare [`ANSWER_OPEN`] was stripped, so the next `\n` is framing. Only
    /// the prefilled-prompt path needs this; every delimiter-driven section is
    /// handled inside the shared machine.
    pending_answer_open_newline: bool,
    /// Content buffered until the stream has settled whether it opens with a
    /// bare [`ANSWER_OPEN`]. `None` once that question is answered.
    answer_open_probe: Option<String>,
    /// Content suffix that could still grow into [`ANSWER_END`].
    held_content: String,
}

impl HunyuanA13BReasoningParser {
    /// Create a Hunyuan A13B parser backed by the shared delimited state
    /// machine.
    ///
    /// Unlike its siblings this cannot fail: the delimiters are matched as text
    /// and never resolved against the vocabulary.
    pub fn new(tokenizer: DynTokenizer) -> Self {
        Self {
            inner: DelimitedReasoningParser::new_text_only(
                tokenizer,
                THINK_START,
                ANSWER_START,
                false,
            )
            .strip_framing_newlines(),
            pending_answer_open_newline: false,
            answer_open_probe: Some(String::new()),
            held_content: String::new(),
        }
    }

    /// Drop a prefilled `<answer>` marker and suppress the answer terminator.
    fn process(&mut self, mut delta: ReasoningDelta) -> ReasoningDelta {
        if let Some(content) = delta.content.take()
            && let Some(mut content) = self.strip_leading_answer_open(content)
        {
            if self.pending_answer_open_newline {
                self.pending_answer_open_newline = false;
                if content.starts_with('\n') {
                    content.remove(0);
                }
            }
            delta.content = self.suppress_answer_end(&content);
        }

        delta
    }

    /// Drop a bare [`ANSWER_OPEN`] opening the stream.
    ///
    /// With `enable_thinking is false` the chat template prefills
    /// `<think>\n\n</think>\n` and stops there, so the model body opens with an
    /// `<answer>` that never reaches the shared machine as part of a delimiter.
    /// It is protocol framing either way. The marker can be split across pushes,
    /// so it needs its own hold-back: the shared machine releases `<a` as soon as
    /// it stops being a possible `<think>`.
    fn strip_leading_answer_open(&mut self, content: String) -> Option<String> {
        let Some(probe) = self.answer_open_probe.as_mut() else {
            return Some(content);
        };
        probe.push_str(&content);

        if let Some(rest) = probe.strip_prefix(ANSWER_OPEN) {
            let rest = rest.to_string();
            self.answer_open_probe = None;
            // Whatever follows the marker opens the answer section, so its
            // first newline is framing just like the delimited path's.
            self.pending_answer_open_newline = true;
            return (!rest.is_empty()).then_some(rest);
        }

        if ANSWER_OPEN.starts_with(probe.as_str()) {
            return None;
        }

        let settled = std::mem::take(probe);
        self.answer_open_probe = None;
        Some(settled)
    }

    /// Strip [`ANSWER_END`] from content, holding back any trailing suffix that
    /// could still grow into it.
    fn suppress_answer_end(&mut self, content: &str) -> Option<String> {
        self.held_content.push_str(content);

        let mut emitted = String::new();
        // Only the marker itself is dropped; a stream that opens another
        // `<think>` section afterwards keeps being parsed.
        while let Some(idx) = self.held_content.find(ANSWER_END) {
            emitted.push_str(&self.held_content[..idx]);
            self.held_content.drain(..idx + ANSWER_END.len());
        }

        let stable_len = self.held_content.len() - partial_answer_end_len(&self.held_content);
        emitted.push_str(&self.held_content[..stable_len]);
        self.held_content.drain(..stable_len);

        (!emitted.is_empty()).then_some(emitted)
    }
}

impl ReasoningParser for HunyuanA13BReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids);
        self.pending_answer_open_newline = false;
        self.answer_open_probe = Some(String::new());
        self.held_content.clear();
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        let inner_delta = self.inner.push(delta);
        Ok(self.process(inner_delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let inner_delta = self.inner.finish();
        let mut delta = self.process(inner_delta);

        // Nothing more can arrive, so an unresolved `<answer>` probe was
        // ordinary content. It is necessarily the only content in this delta:
        // while the probe holds, `process` emits none.
        if let Some(probe) = self.answer_open_probe.take() {
            delta.push_content(&probe);
        }

        // An unfinished `\n</answer>` was never framing, so emit it as content
        // rather than swallowing it at end of stream.
        let held = std::mem::take(&mut self.held_content);
        delta.push_content(&held);

        Ok(delta)
    }
}

/// Return the longest trailing suffix that could still complete [`ANSWER_END`].
fn partial_answer_end_len(text: &str) -> usize {
    text.char_indices()
        .map(|(idx, _)| idx)
        .find(|idx| {
            let suffix = &text[*idx..];
            ANSWER_END.starts_with(suffix) && ANSWER_END != suffix
        })
        .map_or(0, |idx| text.len() - idx)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::HunyuanA13BReasoningParser;
    use crate::reasoning::{ReasoningDelta, ReasoningParser};

    /// The behavioural spec, taken from
    /// `tests/reasoning/test_hunyuan_reasoning_parser.py`.
    const PYTHON_CASES: &[(&str, &str)] = &[
        (
            "simple",
            "<think>\nThis is a reasoning section\n</think>\n<answer>\nThis is the rest\n</answer>",
        ),
        (
            "complete reasoning, no answer body",
            "<think>\nThis is a reasoning section\n</think>\n<answer>\n",
        ),
        (
            "quick thought, empty reasoning",
            "<think>\n\n</think>\n<answer>\nThis is the rest\n</answer>",
        ),
        ("no markers at all", "This is content"),
        (
            "multiple lines, unterminated answer",
            "<think>\nThis\nThat\n</think>\n<answer>\nThis is the rest\nThat",
        ),
    ];

    /// Cases the Python suite never exercises, where this parser deliberately
    /// does something better than the Python one.
    const ADAPTED_CASES: &[(&str, &str)] = &[
        (
            "prefilled prompt, body opens at <answer>",
            "<answer>\nThis is the rest\n</answer>",
        ),
        (
            "content that only looks like the opener",
            "<abc>This is the rest",
        ),
        (
            "text following a complete answer terminator",
            "<think>\nreason\n</think>\n<answer>\nanswer\n</answer>trailing",
        ),
        (
            // Python's state machine cycles response -> idle -> think, so a
            // second round trip is a supported shape. Every section's framing
            // newline has to go, not just the first one's.
            "two complete round trips",
            "<think>\na\n</think>\n<answer>\nb\n</answer><think>\nc\n</think>\n<answer>\nd\n</answer>",
        ),
    ];

    /// Build a parser over an empty vocabulary: the delimiters are matched as
    /// text, so no tokenizer entries are needed.
    fn parser() -> HunyuanA13BReasoningParser {
        HunyuanA13BReasoningParser::new(Arc::new(TestTokenizer::new()))
    }

    /// Push every chunk, flush, and drop the deltas that carry no text.
    fn collect<S: AsRef<str>>(chunks: &[S]) -> Vec<ReasoningDelta> {
        collect_with(parser(), chunks)
    }

    /// Push every chunk into an already-initialized parser, then flush.
    fn collect_with<S: AsRef<str>>(
        mut parser: HunyuanA13BReasoningParser,
        chunks: &[S],
    ) -> Vec<ReasoningDelta> {
        let mut deltas: Vec<_> =
            chunks.iter().map(|chunk| parser.push(chunk.as_ref()).unwrap()).collect();
        deltas.push(parser.finish().unwrap());
        deltas.retain(|delta| !delta.is_empty());
        deltas
    }

    /// Concatenate the reasoning and content of every delta in a stream.
    fn joined<S: AsRef<str>>(chunks: &[S]) -> (Option<String>, Option<String>) {
        let mut reasoning: Option<String> = None;
        let mut content: Option<String> = None;
        for delta in collect(chunks) {
            for (part, sink) in [
                (delta.reasoning, &mut reasoning),
                (delta.content, &mut content),
            ] {
                if let Some(text) = part {
                    sink.get_or_insert_default().push_str(&text);
                }
            }
        }
        (reasoning, content)
    }

    /// Split `output` into chunks of at most `size` characters.
    fn split(output: &str, size: usize) -> Vec<String> {
        output
            .chars()
            .collect::<Vec<_>>()
            .chunks(size)
            .map(|chunk| chunk.iter().collect())
            .collect()
    }

    #[test]
    fn splits_every_python_case() {
        let parsed: Vec<_> =
            PYTHON_CASES.iter().map(|(name, output)| (name, joined(&[output]))).collect();

        expect![[r#"
            [
                (
                    "simple",
                    (
                        Some(
                            "This is a reasoning section",
                        ),
                        Some(
                            "This is the rest",
                        ),
                    ),
                ),
                (
                    "complete reasoning, no answer body",
                    (
                        Some(
                            "This is a reasoning section",
                        ),
                        None,
                    ),
                ),
                (
                    "quick thought, empty reasoning",
                    (
                        None,
                        Some(
                            "This is the rest",
                        ),
                    ),
                ),
                (
                    "no markers at all",
                    (
                        None,
                        Some(
                            "This is content",
                        ),
                    ),
                ),
                (
                    "multiple lines, unterminated answer",
                    (
                        Some(
                            "This\nThat",
                        ),
                        Some(
                            "This is the rest\nThat",
                        ),
                    ),
                ),
            ]
        "#]]
        .assert_debug_eq(&parsed);
    }

    #[test]
    fn streaming_matches_single_push_for_every_python_case() {
        // Chunk boundaries land inside the delimiters at these sizes, which is
        // what the inner state machine's partial-suffix hold-back is for.
        for (name, output) in PYTHON_CASES {
            let whole = joined(&[output]);
            for size in [1, 2, 3, 5, 7, 11] {
                assert_eq!(
                    joined(&split(output, size)),
                    whole,
                    "{name}, chunk size {size}"
                );
            }
        }
    }

    #[test]
    fn holds_answer_start_split_across_pushes() {
        expect![[r#"
            [
                ReasoningDelta {
                    reasoning: Some(
                        "reason",
                    ),
                    content: None,
                },
                ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        "answer",
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&collect(&[
            "<think>\nreason\n</think>\n<ans",
            "wer>\nanswer\n</answer>",
        ]));
    }

    #[test]
    fn emits_incomplete_answer_end_at_finish() {
        // `\n</ans` is held back in case `\n</answer>` completes, then released
        // once the stream ends: it was ordinary content all along.
        expect![[r#"
            [
                ReasoningDelta {
                    reasoning: Some(
                        "reason",
                    ),
                    content: Some(
                        "answer",
                    ),
                },
                ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        "\n</ans",
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&collect(&[
            "<think>\nreason\n</think>\n<answer>\nanswer",
            "\n</ans",
        ]));
    }

    #[test]
    fn ignores_the_prompt_reasoning_boundary() {
        // With no delimiter IDs to resolve there is no prompt boundary to read,
        // so initialization always leaves the parser outside reasoning and the
        // stream's own `<think>` is what opens it.
        let mut parser = parser();
        parser.initialize(&[1, 2, 3]).unwrap();

        expect![[r#"
            [
                ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        "reason\n</think>\n<answer>\nanswer",
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&collect_with(
            parser,
            &["reason\n</think>\n<answer>\nanswer"],
        ));
    }

    #[test]
    fn adapts_where_the_python_parser_leaks_framing() {
        let parsed: Vec<_> =
            ADAPTED_CASES.iter().map(|(name, output)| (name, joined(&[output]))).collect();

        expect![[r#"
            [
                (
                    "prefilled prompt, body opens at <answer>",
                    (
                        None,
                        Some(
                            "This is the rest",
                        ),
                    ),
                ),
                (
                    "content that only looks like the opener",
                    (
                        None,
                        Some(
                            "<abc>This is the rest",
                        ),
                    ),
                ),
                (
                    "text following a complete answer terminator",
                    (
                        Some(
                            "reason",
                        ),
                        Some(
                            "answertrailing",
                        ),
                    ),
                ),
                (
                    "two complete round trips",
                    (
                        Some(
                            "ac",
                        ),
                        Some(
                            "bd",
                        ),
                    ),
                ),
            ]
        "#]]
        .assert_debug_eq(&parsed);
    }

    #[test]
    fn streaming_matches_single_push_for_adapted_cases() {
        for (name, output) in ADAPTED_CASES {
            let whole = joined(&[output]);
            for size in [1, 2, 3, 5, 7, 11] {
                assert_eq!(
                    joined(&split(output, size)),
                    whole,
                    "{name}, chunk size {size}"
                );
            }
        }
    }

    #[test]
    fn emits_an_unresolved_answer_opener_at_finish() {
        // `<ans` is held back in case `<answer>` completes, then released once
        // the stream ends: it was ordinary content all along.
        expect![[r#"
            [
                ReasoningDelta {
                    reasoning: None,
                    content: Some(
                        "<ans",
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&collect(&["<ans"]));
    }
}
