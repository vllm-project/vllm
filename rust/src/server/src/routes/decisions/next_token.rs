// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Next-token reads: each question is one request whose reply's first token
//! is read for the logprobs of the option labels.
//!
//! Original Python definition: `NextTokenStrategy` in
//! `vllm/entrypoints/openai/decisions/strategies.py`.

use std::collections::HashMap;
use std::num::NonZeroU32;

use futures::future;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_chat::{ChatLlm, ChatMessage, ChatOptions, ChatRequest, ChatRole, ResolvedToolContext};
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_text::tokenizer::Tokenizer;
use vllm_text::{FinishReason, Prompt, TextDecodeOptions, TextOutputStreamExt as _};

use super::questions::{Question, QuestionType, label_softmax};
use crate::error::{
    ApiError, bail_invalid_request, chat_submit_error, engine_error, server_error,
    text_submit_error,
};
use crate::routes::generative_scoring::label_logprobs;

/// The most questions one request may ask.
pub(super) const MAX_QUESTIONS: usize = 64;

/// What every read's prompt ends with, and the token each label adds after it
/// as the first token of the reply. Probed once at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NextTokenReads {
    /// The generation prompt's token IDs after its last added token.
    tail: Vec<u32>,
    /// The reply's first token for each label of every question type.
    label_ids: HashMap<&'static str, u32>,
}

/// One question's answer distribution and token usage.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct QuestionRead {
    /// The probability of each label among the labels; sums to 1.
    pub probs: Vec<f64>,
    /// The labels' total probability over the whole vocabulary.
    pub label_mass: f64,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cached_tokens: usize,
    pub cache_write_tokens: usize,
    /// Whether the greedy reply's first token is one of the labels.
    pub argmax_is_label: bool,
}

/// Per-request settings shared by a decision's reads.
#[derive(Default)]
pub(super) struct ReadOptions {
    pub request_id: String,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub lora_request: Option<LoraRequest>,
    pub priority: i32,
    pub cache_salt: Option<String>,
}

impl NextTokenReads {
    /// Probe the chat template: every label of every question type must be
    /// one token, distinct within its type, as the first token of the reply.
    /// Fails otherwise, and the decision endpoints stay off.
    pub(crate) fn probe(chat: &ChatLlm) -> Result<Self, String> {
        let request = chat_request(
            "decisions-probe".to_owned(),
            vec![ChatMessage::text(ChatRole::User, "x")],
            thinking_off(None),
            &ReadOptions::default(),
        );
        let rendered = (chat.request_processor().render(&request))
            .map_err(|error| error.to_report_string())?;
        let tokenizer = chat.text().tokenizer();
        let prompt = match rendered.prompt {
            Prompt::Text(text) => {
                tokenizer.encode(&text, false).map_err(|e| e.to_report_string())?
            }
            Prompt::TokenIds(token_ids) => token_ids,
        };
        let tail = prompt_tail(&*tokenizer, &prompt);
        let text = tokenizer.decode(&tail, false).map_err(|e| e.to_report_string())?;

        let mut label_ids = HashMap::new();
        for question_type in QuestionType::ALL {
            let mut type_ids = Vec::new();
            for &label in question_type.label_set() {
                let extended = (tokenizer.encode(&format!("{text}{label}"), false))
                    .map_err(|e| e.to_report_string())?;
                match extended.split_last() {
                    Some((&id, rest)) if rest == tail && !type_ids.contains(&id) => {
                        type_ids.push(id);
                        label_ids.insert(label, id);
                    }
                    _ => {
                        return Err(format!(
                            "label '{label}' is not one distinct token after this model's \
                             chat prompt"
                        ));
                    }
                }
            }
        }
        Ok(Self { tail, label_ids })
    }

    /// Read each question as one request: the state, then the question, as a
    /// user turn, with the label tokens' logprobs as the reply's first token.
    pub(super) async fn read(
        &self,
        chat: &ChatLlm,
        questions: &[Question],
        instructions: Option<&str>,
        state: &str,
        options: ReadOptions,
    ) -> Result<Vec<QuestionRead>, ApiError> {
        if (options.chat_template_kwargs.as_ref())
            .and_then(|kwargs| kwargs.get("enable_thinking"))
            .is_some_and(is_truthy)
        {
            bail_invalid_request!("a read is the reply's first token, so thinking must be off");
        }
        let template_kwargs = thinking_off(options.chat_template_kwargs.clone());
        let text = chat.text();

        // Prepare every read before submitting any, so a request that fails
        // validation leaves nothing in flight.
        let mut prepared = Vec::with_capacity(questions.len());
        let mut question_label_ids = Vec::with_capacity(questions.len());
        for (index, question) in questions.iter().enumerate() {
            let label_ids: Vec<u32> =
                (question.labels().iter()).map(|label| self.label_ids[label]).collect();
            let mut messages = Vec::with_capacity(2);
            if let Some(instructions) = instructions.filter(|i| !i.is_empty()) {
                messages.push(ChatMessage::text(ChatRole::System, instructions));
            }
            let content = format!("{state}\n\n{}", question.prompt());
            messages.push(ChatMessage::text(ChatRole::User, content));
            let mut request = chat_request(
                format!("{}-{index}", options.request_id),
                messages,
                template_kwargs.clone(),
                &options,
            );
            request.sampling_params = sampling_params(&label_ids);

            let mut text_request = (chat.request_processor().prepare_for_tokenization(request))
                .await
                .map_err(|error| chat_submit_error("failed to render a decision read", error))?;
            let prompt_token_ids = (text.request_processor().tokenize_in_place(&mut text_request))
                .map_err(|error| text_submit_error("failed to tokenize a decision read", error))?;
            if !prompt_token_ids.ends_with(&self.tail) {
                bail_invalid_request!(
                    "these chat options end the prompt differently, so the labels' tokens are \
                     unknown"
                );
            }
            prepared.push(
                text.request_processor().prepare(text_request).map_err(|error| {
                    text_submit_error("failed to prepare a decision read", error)
                })?,
            );
            question_label_ids.push(label_ids);
        }

        // TODO: on hybrid models with prefix caching, prefill the prompts'
        // common prefix (at least 128 tokens) once before the reads, like
        // Python's `NextTokenStrategy._warm_shared_prefix`. Reads that start
        // together cannot reuse each other's recurrent state, so each read
        // prefills it again and reports fewer `cached_tokens` than Python.
        let outputs = future::try_join_all(prepared.into_iter().map(|prepared| async {
            let mut streams = (text.generate_prepared_n(prepared, NonZeroU32::MIN).await)
                .map_err(|error| text_submit_error("failed to submit a decision read", error))?;
            let stream = streams.pop().expect("one output stream per sample");
            stream.collect_output().await.map_err(|error| {
                server_error!(
                    "failed to collect a decision read: {}",
                    error.to_report_string()
                )
            })
        }))
        .await?;

        let mut reads = Vec::with_capacity(outputs.len());
        for (index, (output, label_ids)) in outputs.iter().zip(&question_label_ids).enumerate() {
            // Like Python, an engine error fails the whole decision.
            if let FinishReason::Error(error) = &output.finish_reason {
                return Err(engine_error(error.as_deref()));
            }
            let logprobs = label_logprobs(output, label_ids, index)?;
            let usage = &output.usage;
            let argmax_is_label =
                (output.token_ids.first()).is_some_and(|id| label_ids.contains(id));
            reads.push(QuestionRead {
                probs: label_softmax(&logprobs),
                label_mass: logprobs.iter().map(|logprob| logprob.exp()).sum(),
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.output_token_count,
                cached_tokens: usage.prompt_cache_usage.cached_token_count,
                cache_write_tokens: usage.prompt_cache_usage.cache_creation_token_count,
                argmax_is_label,
            });
        }
        Ok(reads)
    }
}

/// One greedy step that returns the label tokens' logprobs, Python's
/// `SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=...)`.
/// The model's generation config does not apply. `logprobs` stays unset, so
/// a choice may have more labels than `--max-logprobs`.
fn sampling_params(label_ids: &[u32]) -> vllm_text::SamplingParams {
    vllm_text::SamplingParams {
        max_tokens: Some(1),
        temperature: Some(0.0),
        logprob_token_ids: Some(label_ids.to_vec()),
        ..vllm_text::SamplingParams::neutral()
    }
}

/// The request's chat template kwargs over thinking turned off, since the
/// label is the reply's first token.
fn thinking_off(kwargs: Option<HashMap<String, Value>>) -> HashMap<String, Value> {
    let mut merged = HashMap::from([("enable_thinking".to_owned(), Value::Bool(false))]);
    // Like Python's `merge_kwargs`, `null` and `"auto"` leave a value unset.
    merged.extend(
        (kwargs.unwrap_or_default().into_iter())
            .filter(|(_, value)| !value.is_null() && value.as_str() != Some("auto")),
    );
    merged
}

/// Python's truthiness of a JSON value.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// A chat request that ends at the generation prompt, tokenized without
/// special tokens.
fn chat_request(
    request_id: String,
    messages: Vec<ChatMessage>,
    mut template_kwargs: HashMap<String, Value>,
    options: &ReadOptions,
) -> ChatRequest {
    let tool_context = ResolvedToolContext::new(&messages, Vec::new(), None, true)
        .expect("plain text messages declare no tools");
    // Like Python, a `chat_template` kwarg replaces the model's template, which
    // the renderer refuses without `--trust-request-chat-template`.
    let chat_template = match template_kwargs.remove("chat_template") {
        Some(Value::String(template)) => Some(template),
        Some(other) => {
            template_kwargs.insert("chat_template".to_owned(), other);
            None
        }
        None => None,
    };
    ChatRequest {
        request_id,
        messages,
        sampling_params: vllm_text::SamplingParams::default(),
        chat_options: ChatOptions {
            chat_template,
            template_kwargs,
            ..ChatOptions::default()
        },
        tool_context,
        // The sampled token is never read, so it is not decoded either.
        decode_options: TextDecodeOptions {
            detokenize: false,
            ..TextDecodeOptions::default()
        },
        intermediate: false,
        prompt_truncation: None,
        priority: options.priority,
        documents: None,
        cache_salt: options.cache_salt.clone(),
        add_special_tokens: false,
        // Like Python, reads run on any engine and outside any session.
        data_parallel_rank: None,
        session_id: None,
        lora_request: options.lora_request.clone(),
        return_prompt_text: false,
    }
}

/// The token IDs after the prompt's last added token.
fn prompt_tail(tokenizer: &dyn Tokenizer, prompt: &[u32]) -> Vec<u32> {
    let added = tokenizer.added_vocab();
    let start = (prompt.iter())
        .rposition(|token_id| added.iter().any(|(_, id)| id == token_id))
        .map_or(0, |position| position + 1);
    prompt[start..].to_vec()
}
