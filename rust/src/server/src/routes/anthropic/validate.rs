// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;

use super::types::{
    ContextEdit, ContextManagementInput, CountTokensRequest, MessagesInput, MessagesRequest,
    ThinkingConfig,
};
use crate::error::{ApiError, bail_invalid_request};
use crate::lora::LoraModelResolution;

pub(super) fn validate_messages_request(request: &MessagesRequest) -> Result<(), ApiError> {
    if request.temperature.is_some_and(|temperature| temperature > 1.0) {
        bail_invalid_request!(param = "temperature", "Temperature must be in [0, 1]");
    }
    reject_unsupported_fields(request)
}

fn reject_unsupported_fields(request: &MessagesRequest) -> Result<(), ApiError> {
    for (field, supplied) in [
        (
            "output_config.format",
            request.output_config.as_ref().is_some_and(|config| config.format.is_some()),
        ),
        ("output_format", request.output_format.is_some()),
        ("kv_transfer_params", request.kv_transfer_params.is_some()),
        ("ec_transfer_params", request.ec_transfer_params.is_some()),
    ] {
        if supplied {
            bail_invalid_request!(param = field, "{field} is not supported");
        }
    }
    Ok(())
}

pub(super) fn validate_count_tokens_request(request: &CountTokensRequest) -> Result<(), ApiError> {
    if matches!(
        request.input.thinking.as_ref(),
        Some(ThinkingConfig::Enabled {
            budget_tokens: None,
            ..
        })
    ) {
        bail_invalid_request!(
            param = "thinking.budget_tokens",
            "thinking.budget_tokens is required when thinking is enabled"
        );
    }
    Ok(())
}

pub(super) fn validate_messages_input(
    input: &MessagesInput,
    model: &LoraModelResolution,
) -> Result<(), ApiError> {
    validate_context_edits(input.context_management.as_ref())?;
    if input.model.is_empty() || input.messages.is_empty() {
        bail_invalid_request!("model and messages must be non-empty");
    }
    model.model_names.check(Some(&input.model))?;
    if let Some(ThinkingConfig::Enabled {
        budget_tokens: Some(budget_tokens),
        ..
    }) = input.thinking.as_ref()
        && *budget_tokens < 1024
    {
        bail_invalid_request!(
            param = "thinking.budget_tokens",
            "thinking.budget_tokens must be at least 1024"
        );
    }
    Ok(())
}

fn validate_context_edits(context: Option<&ContextManagementInput>) -> Result<(), ApiError> {
    let edits = context.and_then(|context| context.edits.as_deref()).unwrap_or_default();
    // Claude Code explicitly asks to preserve all thinking in the supplied history.
    if edits.iter().any(|edit| {
        !matches!(edit, ContextEdit::ClearThinking { keep }
            if keep.as_ref().and_then(Value::as_str) == Some("all"))
    }) {
        bail_invalid_request!(
            param = "context_management",
            "Context editing is not supported"
        );
    }
    Ok(())
}
