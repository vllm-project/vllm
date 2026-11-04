// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Merging the per-choice streams of a request with `n > 1` choices.

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

/// Merge the chunk streams of a request's choices, in the order the choices
/// produce them, with each choice's chunks stamped with its index.
///
/// Like Python's streaming response for `n > 1`, the choices' final usages are
/// sent once, as one final chunk: the prompt counts once, completion and
/// reasoning tokens add up, and per-request metrics, which describe a single
/// sample, are reported only for `n = 1`.
#[try_stream]
pub(crate) async fn merge_choice_chunk_streams<C: ChoiceChunk + Send + 'static>(
    choices: Vec<impl Stream<Item = Result<C, ApiError>> + Send + 'static>,
    mut y: TryYielder<C, ApiError>,
) -> Result<(), ApiError> {
    let single_choice = choices.len() == 1;
    let mut chunks = stream::select_all(choices.into_iter().zip(0..).map(|(choice, index)| {
        choice
            .map_ok(move |mut chunk: C| {
                chunk.set_choice_index(index);
                chunk
            })
            .boxed()
    }));
    let mut final_usage_chunk: Option<C> = None;

    while let Some(chunk) = chunks.next().await {
        let mut chunk = chunk?;
        let Some(usage) = chunk.final_usage_mut() else {
            y.yield_ok(chunk).await;
            continue;
        };
        match final_usage_chunk.as_mut().and_then(C::final_usage_mut) {
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
