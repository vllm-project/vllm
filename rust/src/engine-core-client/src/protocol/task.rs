// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

/// Generation task supported by the model runner.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/tasks.py#L7>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationTask {
    /// Generate text from an input prompt.
    Generate,
    /// Transcribe or translate audio input.
    Transcription,
    /// Run realtime audio inference.
    Realtime,
}

/// Pooling task supported by the model runner.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/tasks.py#L10-L17>
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

/// Task reported by EngineCore's `get_supported_tasks` utility.
///
/// EngineCore reports the generation and pooling tasks of its model runner.
/// Frontend-only tasks such as `render` never come from the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EngineTask {
    /// A generation task.
    Generation(GenerationTask),
    /// A pooling task.
    Pooling(PoolingTask),
}

impl From<GenerationTask> for EngineTask {
    fn from(task: GenerationTask) -> Self {
        Self::Generation(task)
    }
}

impl From<PoolingTask> for EngineTask {
    fn from(task: PoolingTask) -> Self {
        Self::Pooling(task)
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::EngineTask;

    #[test]
    fn engine_tasks_round_trip_python_wire_literals() {
        let literals = [
            "generate",
            "transcription",
            "realtime",
            "embed",
            "classify",
            "token_embed",
            "token_classify",
            "plugin",
            "embed&token_classify",
        ];
        let encoded = rmp_serde::to_vec(&literals).unwrap();
        let tasks: Vec<EngineTask> = rmp_serde::from_slice(&encoded).unwrap();

        expect![[r#"
            [
                Generation(
                    Generate,
                ),
                Generation(
                    Transcription,
                ),
                Generation(
                    Realtime,
                ),
                Pooling(
                    Embed,
                ),
                Pooling(
                    Classify,
                ),
                Pooling(
                    TokenEmbed,
                ),
                Pooling(
                    TokenClassify,
                ),
                Pooling(
                    Plugin,
                ),
                Pooling(
                    EmbedAndTokenClassify,
                ),
            ]
        "#]]
        .assert_debug_eq(&tasks);

        let reencoded: Vec<String> =
            rmp_serde::from_slice(&rmp_serde::to_vec(&tasks).unwrap()).unwrap();
        assert_eq!(reencoded, literals);
    }

    #[test]
    fn removed_pooling_task_is_a_decode_error() {
        // `score` was removed from Python's `PoolingTask`; an engine never
        // reports it, so it must not decode as some other task.
        let encoded = rmp_serde::to_vec(&["score"]).unwrap();
        assert!(rmp_serde::from_slice::<Vec<EngineTask>>(&encoded).is_err());
    }
}
