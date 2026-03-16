use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use smg_tokenizer::TokenizerTrait as SmgTokenizerTrait;
use smg_tokenizer::chat_template::ChatTemplateParams;
use smg_tokenizer::factory::create_tokenizer_async;
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;

/// [`Tokenizer`] implementation backed by the crates.io `llm-tokenizer` package, imported here as
/// `smg_tokenizer`.
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
    /// Load a tokenizer and any adjacent/default chat template for one model or local path.
    pub async fn from_model_or_path(model_name_or_path: &str) -> Result<Self> {
        let inner = create_tokenizer_async(model_name_or_path)
            .await
            .map_err(|error| Error::Tokenizer(error.to_report_string()))?;

        Ok(Self { inner })
    }

    /// Wrap an existing tokenizer trait object.
    pub fn from_arc(inner: Arc<dyn SmgTokenizerTrait>) -> Self {
        Self { inner }
    }

    pub(crate) fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        continue_final_message: bool,
        template_kwargs: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        let mut merged_template_kwargs = template_kwargs.cloned().unwrap_or_default();
        merged_template_kwargs.insert(
            "continue_final_message".to_string(),
            serde_json::Value::Bool(continue_final_message),
        );

        self.inner
            .apply_chat_template(
                messages,
                ChatTemplateParams {
                    add_generation_prompt,
                    tools: None,
                    documents: None,
                    template_kwargs: Some(&merged_template_kwargs),
                },
            )
            .map_err(|error| Error::Tokenizer(error.to_report_string()))
    }

    /// Return whether the resolved chat template expects plain string message content.
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
