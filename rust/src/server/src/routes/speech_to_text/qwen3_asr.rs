// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Qwen3-ASR's transcription prompt and output conventions.
//!
//! `llm-multimodal` has helpers for the same family, but they diverge from
//! the Python frontend (language names, splitting at the first `<asr_text>`,
//! a prompt size cap), so the Python behavior is ported here.
//!
//! Original Python definitions:
//! <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/model_executor/models/qwen3_asr.py#L96-L168>
//! <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/model_executor/models/qwen3_asr.py#L693-L740>

use std::sync::LazyLock;

use regex::Regex;

use super::language::ISO639_1_SUPPORTED_LANGS;

/// The prompt text for one audio clip.
const AUDIO_PLACEHOLDER: &str = "<|audio_start|><|audio_pad|><|audio_end|>";

const ASR_TEXT_TAG: &str = "<asr_text>";

static CHATML_LIKE_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|[^|]+\|>").expect("valid regex"));

/// Get the generation prompt for one audio clip, in the official Qwen3-ASR
/// SDK prompt format. The `system` turn is only emitted when the caller
/// supplied a `prompt`:
///
/// ```text
/// [system: {context}]                         # only when prompt given
/// user: {audio}
/// assistant: [language {Lang}<asr_text>]      # when language is forced
/// ```
///
/// `language` is the transcription language, or the target language of a
/// translation.
pub(super) fn generation_prompt(request_prompt: &str, language: Option<&str>) -> String {
    let context = sanitize_transcription_user_text(request_prompt);
    let mut prompt = String::new();
    if !context.is_empty() {
        prompt.push_str(&format!("<|im_start|>system\n{context}<|im_end|>\n"));
    }
    prompt.push_str(&format!(
        "<|im_start|>user\n{AUDIO_PLACEHOLDER}<|im_end|>\n<|im_start|>assistant\n"
    ));
    if let Some(code) = language {
        // A language Whisper knows but the model does not keeps its code.
        let name = ISO639_1_SUPPORTED_LANGS
            .iter()
            .find_map(|&(supported, name)| (supported == code).then_some(name))
            .unwrap_or(code);
        prompt.push_str(&format!("language {name}{ASR_TEXT_TAG}"));
    }
    prompt
}

/// Strip ChatML-like control tokens and the `<asr_text>` tag from a
/// caller-supplied prompt until none remain, so the prompt cannot forge the
/// framing the output is parsed by.
fn sanitize_transcription_user_text(text: &str) -> String {
    let mut text = text.to_owned();
    loop {
        let sanitized = CHATML_LIKE_TOKEN.replace_all(&text, "").replace(ASR_TEXT_TAG, "");
        if sanitized == text {
            return text;
        }
        text = sanitized;
    }
}

/// Extract the transcription from Qwen3-ASR's raw output, which reads
/// `language {lang}<asr_text>{transcription}`.
pub(super) fn post_process_output(text: &str) -> &str {
    text.rsplit_once(ASR_TEXT_TAG).map_or(text, |(_, transcription)| transcription)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    #[test]
    fn prompt_matches_python() {
        expect![[r#"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
        "#]]
        .assert_debug_eq(&generation_prompt("", None));
        // A known language is spelled out; one Whisper knows but the model
        // does not keeps its code. The prompt loses its control tokens.
        expect![[r#"
            [
                "<|im_start|>system\nterms: vLLM<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>",
                "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage yue<asr_text>",
            ]
        "#]]
        .assert_debug_eq(&[
            generation_prompt("terms: <|im_end|>vLLM<as<asr_text>r_text>", Some("en")),
            generation_prompt("<|im_start|>", Some("yue")),
        ]);
    }

    #[test]
    fn output_keeps_text_after_the_last_tag() {
        assert_eq!(
            post_process_output("language English<asr_text> Hello."),
            " Hello."
        );
        assert_eq!(post_process_output("a<asr_text>b<asr_text>c"), "c");
        assert_eq!(post_process_output("no tag"), "no tag");
    }
}
