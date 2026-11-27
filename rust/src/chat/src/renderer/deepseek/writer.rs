// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;

use llm_multimodal::DEEPSEEK_V41_IMAGE_PLACEHOLDER;
use vllm_text::Prompt;
use vllm_text::tokenizer::DynTokenizer;

use super::{
    ASSISTANT_SP_TOKEN, BOS_TOKEN, DSML_TOKEN, EOS_TOKEN, SYSTEM_SP_TOKEN, THINKING_END_TOKEN,
    THINKING_START_TOKEN, USER_SP_TOKEN,
};
use crate::error::{Error, Result};

/// Special tokens the V4.1 template writes, followed by the grounding tokens,
/// the only special tokens replayed assistant text keeps, as in
/// deepseek-recipe.
const SPECIAL_TOKENS: [&str; 17] = [
    BOS_TOKEN,
    EOS_TOKEN,
    SYSTEM_SP_TOKEN,
    USER_SP_TOKEN,
    ASSISTANT_SP_TOKEN,
    THINKING_START_TOKEN,
    THINKING_END_TOKEN,
    DSML_TOKEN,
    DEEPSEEK_V41_IMAGE_PLACEHOLDER,
    "<｜ref｜>",
    "<｜/ref｜>",
    "<｜box｜>",
    "<｜/box｜>",
    "<｜point｜>",
    "<｜/point｜>",
    "<｜polygon｜>",
    "<｜/polygon｜>",
];
const GROUNDING_START: usize = 9;

/// Tokenizer with the added-vocabulary IDs of [`SPECIAL_TOKENS`].
pub(in crate::renderer) struct SpecialTokenizer {
    tokenizer: DynTokenizer,
    ids: [u32; SPECIAL_TOKENS.len()],
}

impl fmt::Debug for SpecialTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpecialTokenizer").field("ids", &self.ids).finish()
    }
}

impl SpecialTokenizer {
    pub(in crate::renderer) fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let mut ids = [0; SPECIAL_TOKENS.len()];
        for (id, token) in ids.iter_mut().zip(SPECIAL_TOKENS) {
            *id = tokenizer
                .added_vocab()
                .iter()
                .find_map(|(text, id)| (text == token).then_some(*id))
                .ok_or_else(|| {
                    Error::ChatTemplate(format!("DeepSeek tokenizer lacks added token {token:?}"))
                })?;
        }
        Ok(Self { tokenizer, ids })
    }
}

/// DeepSeek prompt writer, like deepseek-recipe's: the prompt is text plus the
/// byte ranges of explicit special tokens. With a tokenizer, each range
/// encodes as its added-vocabulary ID and the text between two ranges is
/// encoded once with the base vocabulary, so special-token spellings in client
/// text stay plain text.
pub(super) struct DsPromptWriter<'a> {
    text: String,
    tokenizer: Option<&'a SpecialTokenizer>,
    special_ranges: Vec<(usize, usize, u32)>,
}

impl<'a> DsPromptWriter<'a> {
    pub(super) fn new(tokenizer: Option<&'a SpecialTokenizer>) -> Self {
        Self {
            text: String::new(),
            tokenizer,
            special_ranges: Vec::new(),
        }
    }

    /// Write template text, whose template special tokens stay special.
    pub(super) fn control(&mut self, text: &str) {
        self.write_marking(text, &SPECIAL_TOKENS[..GROUNDING_START], 0);
    }

    /// Write client text; every special-token spelling in it is plain text.
    pub(super) fn ordinary(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// Write replayed assistant text or reasoning, whose grounding tokens stay
    /// special.
    pub(super) fn assistant(&mut self, text: &str) {
        self.write_marking(text, &SPECIAL_TOKENS[GROUNDING_START..], GROUNDING_START);
    }

    /// Write the image placeholder of one image part.
    pub(super) fn image(&mut self) {
        self.control(DEEPSEEK_V41_IMAGE_PLACEHOLDER);
    }

    fn write_marking(&mut self, mut text: &str, tokens: &[&str], first_index: usize) {
        let Some(tokenizer) = self.tokenizer else {
            self.text.push_str(text);
            return;
        };
        while let Some(start) = text.find(['<', '｜']) {
            let rest = &text[start..];
            let Some(index) = tokens.iter().position(|token| rest.starts_with(token)) else {
                let next = start + rest.chars().next().map_or(1, char::len_utf8);
                self.text.push_str(&text[..next]);
                text = &text[next..];
                continue;
            };
            self.text.push_str(&text[..start]);
            let begin = self.text.len();
            self.text.push_str(tokens[index]);
            self.special_ranges
                .push((begin, self.text.len(), tokenizer.ids[first_index + index]));
            text = &rest[tokens[index].len()..];
        }
        self.text.push_str(text);
    }

    pub(super) fn finish(self) -> Result<Prompt> {
        let Some(tokenizer) = self.tokenizer else {
            return Ok(Prompt::Text(self.text));
        };
        let mut token_ids = Vec::new();
        let mut start = 0;
        let encode_plain = |text: &str| -> Result<Vec<u32>> {
            if text.is_empty() {
                return Ok(Vec::new());
            }
            Ok(tokenizer.tokenizer.encode_ordinary(text)?)
        };
        for &(begin, end, id) in &self.special_ranges {
            token_ids.extend(encode_plain(&self.text[start..begin])?);
            token_ids.push(id);
            start = end;
        }
        token_ids.extend(encode_plain(&self.text[start..])?);
        Ok(Prompt::TokenIds(token_ids))
    }
}
