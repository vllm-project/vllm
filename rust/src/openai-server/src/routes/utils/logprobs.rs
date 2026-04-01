use std::collections::HashMap;

use openai_protocol::common::LogProbs;
use vllm_text::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
};

use super::types::{ChatLogProbs, ChatLogProbsContent, TopLogProb};
use crate::error::{ApiError, server_error};

/// Convert decoded token-position logprobs into the OpenAI completions `logprobs` shape.
pub fn decoded_logprobs_to_openai(
    logprobs: &DecodedLogprobs,
    initial_text_offset: u32,
) -> Result<LogProbs, ApiError> {
    let mut text_offset = Vec::with_capacity(logprobs.positions.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.positions.len());
    let mut tokens = Vec::with_capacity(logprobs.positions.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.positions.len());
    let mut current_offset = initial_text_offset;

    for position in &logprobs.positions {
        let chosen = position.entries.first().ok_or_else(|| {
            server_error!("decoded logprobs position unexpectedly had no token candidates")
        })?;

        text_offset.push(current_offset);
        token_logprobs.push(Some(clamp_logprob(chosen.logprob)));
        tokens.push(chosen.token.clone());
        top_logprobs.push(Some(position_top_logprobs_map(position)));
        current_offset = current_offset.saturating_add(text_len(&chosen.token));
    }

    Ok(LogProbs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    })
}

/// Convert decoded prompt logprobs into the OpenAI completions `logprobs` shape.
///
/// The first prompt token is included with `None` logprob metadata, matching Python vLLM's
/// echoed completions behavior.
pub fn decoded_prompt_logprobs_to_openai(
    prompt_logprobs: &DecodedPromptLogprobs,
    initial_text_offset: u32,
) -> Result<LogProbs, ApiError> {
    let mut text_offset = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut token_logprobs = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut tokens = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut top_logprobs = Vec::with_capacity(prompt_logprobs.scored_positions.len() + 1);
    let mut current_offset = initial_text_offset;

    text_offset.push(current_offset);
    token_logprobs.push(None);
    tokens.push(prompt_logprobs.first_token.clone());
    top_logprobs.push(None);
    current_offset = current_offset.saturating_add(text_len(&prompt_logprobs.first_token));

    for position in &prompt_logprobs.scored_positions {
        let chosen = position.entries.first().ok_or_else(|| {
            server_error!("decoded prompt logprobs position unexpectedly had no token candidates")
        })?;

        text_offset.push(current_offset);
        token_logprobs.push(Some(clamp_logprob(chosen.logprob)));
        tokens.push(chosen.token.clone());
        top_logprobs.push(Some(position_top_logprobs_map(position)));
        current_offset = current_offset.saturating_add(text_len(&chosen.token));
    }

    Ok(LogProbs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    })
}

/// Convert decoded prompt logprobs into the vLLM-style prompt-logprobs response shape.
pub fn decoded_prompt_logprobs_to_maps(
    prompt_logprobs: &DecodedPromptLogprobs,
) -> Vec<Option<HashMap<String, f32>>> {
    std::iter::once(None)
        .chain(
            prompt_logprobs
                .scored_positions
                .iter()
                .map(|position| Some(position_top_logprobs_map(position))),
        )
        .collect()
}

/// Convert decoded token-position logprobs into the OpenAI chat `logprobs` shape.
pub fn decoded_logprobs_to_openai_chat(
    logprobs: &DecodedLogprobs,
) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(position_to_chat_logprobs_content)
        .try_collect()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

/// Count visible text positions using OpenAI completions' character-offset convention.
pub fn text_len(text: &str) -> u32 {
    u32::try_from(text.chars().count()).unwrap_or(u32::MAX)
}

/// Concatenate two OpenAI-style completion logprobs payloads in token order.
pub fn append_openai_logprobs(mut prefix: LogProbs, suffix: LogProbs) -> LogProbs {
    prefix.tokens.extend(suffix.tokens);
    prefix.token_logprobs.extend(suffix.token_logprobs);
    prefix.top_logprobs.extend(suffix.top_logprobs);
    prefix.text_offset.extend(suffix.text_offset);
    prefix
}

/// Build the non-stream completions `logprobs` payload from collected text output.
///
/// When `echoed_prompt` is true, the returned payload matches Python vLLM's echoed completions
/// behavior by concatenating prompt and completion logprobs into one OpenAI `LogProbs` object.
pub fn collected_logprobs_to_openai(
    collected: &CollectedTextOutput,
    echoed_prompt: bool,
    initial_completion_offset: u32,
) -> Result<LogProbs, ApiError> {
    if echoed_prompt {
        let prompt_logprobs = collected.prompt_logprobs.as_ref().ok_or_else(|| {
            server_error!(
                "echoed completion logprobs require prompt logprobs but generation returned none"
            )
        })?;
        let prompt_logprobs = decoded_prompt_logprobs_to_openai(prompt_logprobs, 0)?;
        return match collected.logprobs.as_ref() {
            Some(completion_logprobs) => Ok(append_openai_logprobs(
                prompt_logprobs,
                decoded_logprobs_to_openai(completion_logprobs, initial_completion_offset)?,
            )),
            None => Ok(prompt_logprobs),
        };
    }

    let completion_logprobs = collected.logprobs.as_ref().ok_or_else(|| {
        server_error!("completion response requested logprobs but generation returned none")
    })?;
    decoded_logprobs_to_openai(completion_logprobs, initial_completion_offset)
}

fn position_top_logprobs_map(position: &DecodedPositionLogprobs) -> HashMap<String, f32> {
    position
        .entries
        .iter()
        .map(|entry| (entry.token.clone(), clamp_logprob(entry.logprob)))
        .collect()
}

fn position_to_chat_logprobs_content(
    position: &DecodedPositionLogprobs,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(|| {
        server_error!("decoded chat logprobs position unexpectedly had no token candidates")
    })?;

    Ok(ChatLogProbsContent {
        token: chosen.token.clone(),
        logprob: clamp_logprob(chosen.logprob),
        bytes: Some(token_bytes(&chosen.token)),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| TopLogProb {
                token: entry.token.clone(),
                logprob: clamp_logprob(entry.logprob),
                bytes: Some(token_bytes(&entry.token)),
            })
            .collect(),
    })
}

fn token_bytes(token: &str) -> Vec<u8> {
    token.as_bytes().to_vec()
}

pub fn clamp_logprob(logprob: f32) -> f32 {
    logprob.max(-9999.0)
}
