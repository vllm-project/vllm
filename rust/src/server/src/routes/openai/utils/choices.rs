// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Merging the per-choice streams or responses of a request with several
//! choices: `n > 1`, or a batch of prompts.

use asynk_strim_attr::{TryYielder, try_stream};
use futures::{Stream, StreamExt as _, TryStreamExt as _, stream};

use super::types::Usage;
use crate::error::ApiError;

/// One streamed chunk of a single choice.
pub(crate) trait ChoiceChunk {
    /// Stamp the index of the choice this chunk belongs to.
    fn set_choice_index(&mut self, index: u32);

    /// The usage of the final usage-only chunk, or `None` for any other chunk.
    fn final_usage_mut(&mut self) -> Option<&mut Usage>;

    /// Drop the per-request metrics.
    fn clear_metrics(&mut self);
}

/// The non-streaming response of a single choice.
pub(crate) trait ChoiceResponse {
    /// Stamp the index of the choice this response holds.
    fn set_choice_index(&mut self, index: u32);

    /// The usage of this response, if it reports one.
    fn usage_mut(&mut self) -> Option<&mut Usage>;

    /// Drop the per-request metrics.
    fn clear_metrics(&mut self);

    /// Move the choices of `other` after the choices of this response.
    fn append_choices(&mut self, other: Self);
}

/// Merge the chunk streams of a request's choices, in the order the choices
/// produce them, with each choice's chunks stamped with its index.
///
/// Like Python's streaming response for `n > 1`, the choices' final usages are
/// sent once, as one final chunk: the prompt counts once, completion and
/// reasoning tokens add up, and per-request metrics, which describe a single
/// sample, are reported only for `n = 1`.
pub(crate) fn merge_choice_chunk_streams<C: ChoiceChunk + Send + 'static>(
    choices: Vec<impl Stream<Item = Result<C, ApiError>> + Send + 'static>,
) -> impl Stream<Item = Result<C, ApiError>> + Send + 'static {
    merge_batch_choice_chunk_streams(vec![choices])
}

/// Like [`merge_choice_chunk_streams`], for a batch of prompts: the choices come
/// grouped by prompt, are indexed in prompt-major order, and each prompt counts
/// once in the final usage.
#[try_stream]
pub(crate) async fn merge_batch_choice_chunk_streams<C: ChoiceChunk + Send + 'static>(
    choices_by_prompt: Vec<Vec<impl Stream<Item = Result<C, ApiError>> + Send + 'static>>,
    mut y: TryYielder<C, ApiError>,
) -> Result<(), ApiError> {
    let mut prompt_counted = vec![false; choices_by_prompt.len()];
    let choices =
        (choices_by_prompt.into_iter().enumerate()).flat_map(|(prompt_index, choices)| {
            choices.into_iter().map(move |choice| (prompt_index, choice))
        });
    let mut chunks = stream::select_all(choices.zip(0..).map(|((prompt_index, choice), index)| {
        choice
            .map_ok(move |mut chunk: C| {
                chunk.set_choice_index(index);
                (prompt_index, chunk)
            })
            .boxed()
    }));
    let single_choice = chunks.len() == 1;
    let mut final_usage_chunk: Option<C> = None;

    while let Some(chunk) = chunks.next().await {
        let (prompt_index, mut chunk) = chunk?;
        let Some(usage) = chunk.final_usage_mut() else {
            y.yield_ok(chunk).await;
            continue;
        };
        let count_prompt = !std::mem::replace(&mut prompt_counted[prompt_index], true);
        match final_usage_chunk.as_mut().and_then(C::final_usage_mut) {
            Some(merged) if count_prompt => merged.add_prompt(usage),
            Some(merged) => merged.add_choice(usage),
            None => {
                if !single_choice {
                    chunk.clear_metrics();
                }
                final_usage_chunk = Some(chunk);
            }
        }
    }
    if let Some(chunk) = final_usage_chunk {
        y.yield_ok(chunk).await;
    }
    Ok(())
}

/// Merge the responses of a request's choices, in index order.
///
/// Like Python's non-streaming response for `n > 1`, the prompt-level fields
/// come from the first choice, usage adds up per [`Usage::add_choice`], and
/// per-request metrics, which describe a single sample, are reported only for
/// `n = 1`.
pub(crate) fn merge_choice_responses<R: ChoiceResponse>(responses: Vec<R>) -> R {
    merge_batch_choice_responses(vec![responses])
}

/// Like [`merge_choice_responses`], for a batch of prompts: the choices come
/// grouped by prompt, are indexed in prompt-major order, and each prompt counts
/// once in the usage per [`Usage::add_prompt`].
pub(crate) fn merge_batch_choice_responses<R: ChoiceResponse>(choices_by_prompt: Vec<Vec<R>>) -> R {
    let mut responses = choices_by_prompt.into_iter().flat_map(|choices| {
        (choices.into_iter().enumerate()).map(|(index, response)| (index == 0, response))
    });
    let (_, mut merged) = responses.next().expect("a request has at least one choice");
    for (index, (count_prompt, mut response)) in (1..).zip(responses) {
        merged.clear_metrics();
        if let (Some(usage), Some(other)) = (merged.usage_mut(), response.usage_mut()) {
            if count_prompt {
                usage.add_prompt(other);
            } else {
                usage.add_choice(other);
            }
        }
        response.set_choice_index(index);
        merged.append_choices(response);
    }
    merged
}
