// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::error::ApiError;

/// The public model names a request's `model` field may select.
#[derive(Debug, Clone)]
pub(crate) struct ServedModelNames {
    /// Served base model names. The first entry is the primary name.
    base_names: Vec<String>,
    /// Loaded LoRA adapter names, selectable like the base names.
    adapter_names: Vec<String>,
    /// Accept any other name too, served by the base model, like Python with
    /// `VLLM_SKIP_MODEL_NAME_VALIDATION` (for proxies and gateways that
    /// rewrite or ignore model names).
    accept_any: bool,
}

impl ServedModelNames {
    /// `base_names` must be non-empty; the first entry is the primary name.
    pub fn new(base_names: Vec<String>) -> Self {
        assert!(
            !base_names.is_empty(),
            "served model names must not be empty"
        );
        Self {
            base_names,
            adapter_names: Vec::new(),
            accept_any: false,
        }
    }

    /// Also accept any other name, served by the base model
    /// (`VLLM_SKIP_MODEL_NAME_VALIDATION`).
    pub fn with_accept_any(mut self, accept_any: bool) -> Self {
        self.accept_any = accept_any;
        self
    }

    /// Also accept the names of loaded LoRA adapters.
    pub fn with_adapter_names(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.adapter_names.extend(names);
        self
    }

    /// The name reported as `model` in responses that select no LoRA adapter.
    pub fn primary(&self) -> &str {
        &self.base_names[0]
    }

    /// Served base model names, primary first.
    pub fn base_names(&self) -> &[String] {
        &self.base_names
    }

    /// Whether a request may name `model`. Like Python, an absent or empty
    /// name selects the primary model.
    pub fn accepts(&self, model: Option<&str>) -> bool {
        match model {
            None | Some("") => true,
            Some(model) => {
                self.accept_any
                    || self.base_names.iter().chain(&self.adapter_names).any(|name| name == model)
            }
        }
    }

    /// Reject a model name this server does not serve with a 404 that lists
    /// the base model names, like Python.
    pub fn check(&self, model: Option<&str>) -> Result<(), ApiError> {
        match model {
            Some(model) if !self.accepts(Some(model)) => Err(ApiError::model_not_found(
                model.to_string(),
                self.base_names.clone(),
            )),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ServedModelNames;

    fn served(accept_any: bool) -> ServedModelNames {
        ServedModelNames::new(vec!["base".to_string(), "alias".to_string()])
            .with_accept_any(accept_any)
    }

    #[test]
    fn strict_names_accept_only_served_names() {
        let served = served(false);
        for model in [None, Some(""), Some("base"), Some("alias")] {
            assert!(served.check(model).is_ok(), "{model:?}");
        }
        assert!(served.check(Some("other")).is_err());
    }

    #[test]
    fn unknown_model_lists_base_names_only() {
        let served = served(false).with_adapter_names(["adapter".to_string()]);
        assert!(served.check(Some("adapter")).is_ok());
        let error = served.check(Some("other")).unwrap_err().to_error_response();
        assert_eq!(
            error.error.message,
            "The model `other` does not exist. Valid aliases: base, alias."
        );
    }

    #[test]
    fn skip_validation_accepts_any_name() {
        let served = served(true);
        assert!(served.check(Some("other")).is_ok());
        assert_eq!(served.primary(), "base");
    }
}
