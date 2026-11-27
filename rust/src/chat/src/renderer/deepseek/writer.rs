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

/// Special tokens the V4.1 template writes.
const TEMPLATE_TOKENS: [&str; 9] = [
    BOS_TOKEN,
    EOS_TOKEN,
    SYSTEM_SP_TOKEN,
    USER_SP_TOKEN,
    ASSISTANT_SP_TOKEN,
    THINKING_START_TOKEN,
    THINKING_END_TOKEN,
    DSML_TOKEN,
    DEEPSEEK_V41_IMAGE_PLACEHOLDER,
];

/// Grounding tokens, the only special tokens replayed assistant text keeps,
/// as deepseek-recipe's `ASSISTANT_SPECIAL_TOKENS`.
const GROUNDING_TOKENS: [&str; 8] = [
    "<｜ref｜>",
    "<｜/ref｜>",
    "<｜box｜>",
    "<｜/box｜>",
    "<｜point｜>",
    "<｜/point｜>",
    "<｜polygon｜>",
    "<｜/polygon｜>",
];

/// Tokenizer with the added-vocabulary IDs of [`TEMPLATE_TOKENS`] and
/// [`GROUNDING_TOKENS`].
pub(in crate::renderer) struct SpecialTokenizer {
    tokenizer: DynTokenizer,
    template_ids: [u32; TEMPLATE_TOKENS.len()],
    grounding_ids: [u32; GROUNDING_TOKENS.len()],
}

impl fmt::Debug for SpecialTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpecialTokenizer")
            .field("template_ids", &self.template_ids)
            .field("grounding_ids", &self.grounding_ids)
            .finish()
    }
}

impl SpecialTokenizer {
    pub(in crate::renderer) fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            template_ids: added_token_ids(&tokenizer, TEMPLATE_TOKENS)?,
            grounding_ids: added_token_ids(&tokenizer, GROUNDING_TOKENS)?,
            tokenizer,
        })
    }
}

/// Look up the added-vocabulary ID of each token.
fn added_token_ids<const N: usize>(
    tokenizer: &DynTokenizer,
    tokens: [&str; N],
) -> Result<[u32; N]> {
    let mut ids = [0; N];
    for (id, token) in ids.iter_mut().zip(tokens) {
        *id = tokenizer
            .added_vocab()
            .iter()
            .find_map(|(text, id)| (text == token).then_some(*id))
            .ok_or_else(|| {
                Error::ChatTemplate(format!("DeepSeek tokenizer lacks added token {token:?}"))
            })?;
    }
    Ok(ids)
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
        let ids = self.tokenizer.map(|tokenizer| &tokenizer.template_ids[..]);
        self.write_recognizing(text, &TEMPLATE_TOKENS, ids);
    }

    /// Write client text; every special-token spelling in it is plain text.
    pub(super) fn ordinary(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// Write replayed assistant text or reasoning, whose grounding tokens stay
    /// special.
    pub(super) fn assistant(&mut self, text: &str) {
        let ids = self.tokenizer.map(|tokenizer| &tokenizer.grounding_ids[..]);
        self.write_recognizing(text, &GROUNDING_TOKENS, ids);
    }

    /// Write the image placeholder of one image part.
    pub(super) fn image(&mut self) {
        self.control(DEEPSEEK_V41_IMAGE_PLACEHOLDER);
    }

    /// Write `text`, each spelling of `tokens` in it as a special token whose
    /// ID is at the same index of `ids`.
    fn write_recognizing(&mut self, mut text: &str, tokens: &[&str], ids: Option<&[u32]>) {
        let Some(ids) = ids else {
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
            self.special_ranges.push((begin, self.text.len(), ids[index]));
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
