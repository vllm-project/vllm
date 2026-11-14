// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use vllm_text::tokenizer::DynTokenizer;

use crate::error::Result;

/// Encodes rendered prompts for a tokenizer whose `encode` keeps control
/// tokens as plain text.
///
/// Chat templates spell control tokens out, as in `<s>[INST]hi[/INST]`.
/// Hugging Face tokenizers match them while encoding, but Tekken keeps them as
/// text, as Python's `MistralTokenizer.encode` does. Python renders Mistral
/// chats straight to token IDs instead, so this maps each control token in a
/// rendered prompt to its ID and encodes only the text between them.
pub(super) struct ControlTokenEncoder {
    tokenizer: DynTokenizer,
    /// The tokenizer's added vocabulary, which holds Tekken's control tokens.
    ids: HashMap<String, u32>,
    /// Distinct byte lengths of the control tokens, longest first.
    lens: Vec<usize>,
    /// Whether some control token starts with the given byte.
    first_bytes: [bool; 256],
}

impl ControlTokenEncoder {
    pub(super) fn new(tokenizer: DynTokenizer) -> Self {
        let ids: HashMap<String, u32> = tokenizer
            .added_vocab()
            .iter()
            .filter(|(token, _)| !token.is_empty())
            .cloned()
            .collect();
        let mut lens: Vec<usize> = ids.keys().map(String::len).collect();
        lens.sort_unstable_by(|a, b| b.cmp(a));
        lens.dedup();
        let mut first_bytes = [false; 256];
        for token in ids.keys() {
            first_bytes[usize::from(token.as_bytes()[0])] = true;
        }
        Self {
            tokenizer,
            ids,
            lens,
            first_bytes,
        }
    }

    /// Encode `prompt`, matching the longest control token at each position.
    pub(super) fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut token_ids = Vec::new();
        let mut text_start = 0;
        let mut pos = 0;
        while pos < prompt.len() {
            let Some((len, id)) = self.control_token_at(prompt, pos) else {
                pos += 1;
                continue;
            };
            if text_start < pos {
                token_ids.extend(self.tokenizer.encode_ordinary(&prompt[text_start..pos])?);
            }
            token_ids.push(id);
            pos += len;
            text_start = pos;
        }
        if text_start < prompt.len() {
            token_ids.extend(self.tokenizer.encode_ordinary(&prompt[text_start..])?);
        }
        Ok(token_ids)
    }

    fn control_token_at(&self, prompt: &str, pos: usize) -> Option<(usize, u32)> {
        if !self.first_bytes[usize::from(prompt.as_bytes()[pos])] {
            return None;
        }
        self.lens
            .iter()
            .find_map(|&len| Some((len, *self.ids.get(prompt.get(pos..pos + len)?)?)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_text::tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    #[test]
    fn matches_control_tokens_like_hugging_face_encode() {
        // `TestTokenizer::encode` matches its added tokens as Hugging Face
        // tokenizers do; `encode_ordinary` keeps them as text, as Tekken does.
        let tokenizer = Arc::new(
            TestTokenizer::new()
                .with_special_token("<s>", 300)
                .with_special_token("[INST]", 301)
                .with_special_token("[/INST]", 302)
                .with_special_token("[INST]+", 303),
        );
        let encoder = ControlTokenEncoder::new(tokenizer.clone());

        assert_eq!(
            encoder.encode("<s>[INST]hi[/INST]").unwrap(),
            [300, 301, u32::from(b'h'), u32::from(b'i'), 302]
        );
        for prompt in [
            "",
            "plain text",
            "<s>[INST]Thủ đô 東京 🚀[/INST]",
            "é[INST]é[/INST]é",
            "[INST]+longest match[INST]",
            "[[INST]] <s [INST [/INST",
        ] {
            assert_eq!(
                encoder.encode(prompt).unwrap(),
                tokenizer.encode(prompt, false).unwrap(),
                "{prompt:?}"
            );
        }
    }
}
