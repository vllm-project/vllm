// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_text::Prompt;
use vllm_text::tokenizer::DynTokenizer;

use crate::error::{Result, chat_template_thrown};

/// Control tokens emitted by the GLM5Next template.
const CONTROL_TOKENS: [&str; 24] = [
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
    "<|begin_of_image|>",
    "<|image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|video|>",
    "<|end_of_video|>",
    "<|begin_of_audio|>",
    "<|end_of_audio|>",
];

/// Tokenizer with the added-vocabulary IDs of [`CONTROL_TOKENS`].
pub(super) struct ControlTokenizer {
    tokenizer: DynTokenizer,
    ids: [u32; CONTROL_TOKENS.len()],
}

impl ControlTokenizer {
    pub(super) fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let mut ids = [0; CONTROL_TOKENS.len()];
        for (id, token) in ids.iter_mut().zip(CONTROL_TOKENS) {
            *id = tokenizer
                .added_vocab()
                .iter()
                .find_map(|(text, id)| (text == token).then_some(*id))
                .ok_or_else(|| {
                    chat_template_thrown!("GLM tokenizer lacks added token {token:?}")
                })?;
        }
        Ok(Self { tokenizer, ids })
    }
}

/// GLM prompt writer: control tokens map to added-vocabulary IDs and the text
/// between them is encoded once with the base vocabulary.
pub(super) struct Glm5PromptWriter<'a> {
    text: String,
    tokens: Option<(&'a ControlTokenizer, Vec<u32>)>,
    pending: usize,
}

impl<'a> Glm5PromptWriter<'a> {
    pub(super) fn new(tokenizer: Option<&'a ControlTokenizer>) -> Self {
        Self {
            text: String::new(),
            tokens: tokenizer.map(|tokenizer| (tokenizer, Vec::new())),
            pending: 0,
        }
    }

    pub(super) fn text(&self) -> &str {
        &self.text
    }

    /// Write template text, recognizing the GLM control tokens it contains.
    pub(super) fn control(&mut self, mut text: &str) -> Result<()> {
        if self.tokens.is_none() {
            self.text.push_str(text);
            return Ok(());
        }
        while let Some(start) = text.find(['<', '[']) {
            let rest = &text[start..];
            let Some(index) = CONTROL_TOKENS.iter().position(|token| rest.starts_with(token))
            else {
                self.text.push_str(&text[..=start]);
                text = &text[start + 1..];
                continue;
            };
            self.text.push_str(&text[..start]);
            self.flush()?;
            let (tokenizer, token_ids) = self.tokens.as_mut().unwrap();
            token_ids.push(tokenizer.ids[index]);
            self.text.push_str(CONTROL_TOKENS[index]);
            self.pending = self.text.len();
            text = &rest[CONTROL_TOKENS[index].len()..];
        }
        self.text.push_str(text);
        Ok(())
    }

    /// Write literal text; control-token spellings in it stay plain text.
    pub(super) fn ordinary(&mut self, text: &str) {
        self.text.push_str(text);
    }

    fn flush(&mut self) -> Result<()> {
        if let Some((tokenizer, token_ids)) = &mut self.tokens
            && self.pending < self.text.len()
        {
            let text = &self.text[self.pending..];
            token_ids.extend(tokenizer.tokenizer.encode_ordinary(text)?);
            self.pending = self.text.len();
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Prompt> {
        self.flush()?;
        Ok(match self.tokens {
            Some((_, token_ids)) => Prompt::TokenIds(token_ids),
            None => Prompt::Text(self.text),
        })
    }
}
