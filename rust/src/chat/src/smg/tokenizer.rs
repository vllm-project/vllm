use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use smg_tokenizer::TokenizerTrait as SmgTokenizerTrait;
use smg_tokenizer::chat_template::ChatTemplateParams;
use smg_tokenizer::factory::create_tokenizer_with_chat_template_blocking;
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub struct SmgTokenizer {
    inner: Arc<dyn SmgTokenizerTrait>,
}

impl fmt::Debug for SmgTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmgTokenizer").finish_non_exhaustive()
    }
}

impl SmgTokenizer {
    pub fn from_model_or_path(model_name_or_path: &str) -> Result<Self> {
        Self::from_model_or_path_with_chat_template(model_name_or_path, None::<&Path>)
    }

    pub fn from_model_or_path_with_chat_template(
        model_name_or_path: &str,
        chat_template_path: Option<impl AsRef<Path>>,
    ) -> Result<Self> {
        let chat_template_path = chat_template_path
            .as_ref()
            .map(|path| path.as_ref().to_string_lossy().into_owned());
        let inner = create_tokenizer_with_chat_template_blocking(
            model_name_or_path,
            chat_template_path.as_deref(),
        )
        .map_err(|error| Error::Tokenizer(error.to_report_string()))?;
        Ok(Self { inner })
    }

    pub fn from_arc(inner: Arc<dyn SmgTokenizerTrait>) -> Self {
        Self { inner }
    }

    pub(crate) fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        template_kwargs: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        self.inner
            .apply_chat_template(
                messages,
                ChatTemplateParams {
                    add_generation_prompt,
                    tools: None,
                    documents: None,
                    template_kwargs,
                },
            )
            .map_err(|error| Error::Tokenizer(error.to_report_string()))
    }

    pub(crate) fn supports_string_chat_template(&self) -> bool {
        matches!(
            self.inner.chat_template_content_format(),
            smg_tokenizer::chat_template::ChatTemplateContentFormat::String
        )
    }
}

impl Tokenizer for SmgTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|error| Error::Tokenizer(error.to_report_string()))?;
        Ok(encoding.token_ids().to_vec())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| Error::Tokenizer(error.to_report_string()))
    }
}
