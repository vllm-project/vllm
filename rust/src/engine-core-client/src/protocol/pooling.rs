// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

/// Pooling task executed by the model.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/tasks.py#L10-L17>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingTask {
    /// Produce one embedding for each input sequence.
    Embed,
    /// Produce sequence-level classification outputs.
    Classify,
    /// Produce token-level embeddings.
    TokenEmbed,
    /// Produce token-level classification outputs.
    TokenClassify,
    /// Run a plugin-defined pooling task.
    Plugin,
    /// Produce sequence embeddings and token classifications together.
    #[serde(rename = "embed&token_classify")]
    EmbedAndTokenClassify,
}

/// API parameters for pooling models.
///
/// This is the supported positional prefix of Python's array-like
/// `PoolingParams`. Python fills the omitted internal suffix with its declared
/// defaults.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/pooling_params.py#L38-L73>
///
/// Original Python field documentation:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/config/pooler.py#L51-L115>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct EngineCorePoolingParams {
    /// Whether to apply activation function to the pooler outputs.
    pub use_activation: bool,
    /// Reduce the dimensions of embeddings if model support matryoshka
    /// representation.
    pub dimensions: Option<u32>,
    /// If set, only the score corresponding to the `step_tag_id` in the
    /// generated sentence should be returned. Otherwise, the scores for all
    /// tokens are returned.
    pub step_tag_id: Option<u32>,
    /// A list of indices for the vocabulary dimensions to be extracted,
    /// such as the token IDs of `good_token` and `bad_token` in the
    /// `math-shepherd-mistral-7b-prm` model.
    pub returned_token_ids: Option<Vec<u32>>,
    /// The task used for pooling.
    pub task: PoolingTask,
}
