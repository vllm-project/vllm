// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

/// Generation task supported by the model runner.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/tasks.py#L7-L8>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationTask {
    /// Generate text from an input prompt.
    Generate,
    /// Transcribe audio input.
    Transcription,
    /// Run realtime audio inference.
    Realtime,
}

/// Pooling task supported by the model runner.
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

/// Task supported by an EngineCore model runner.
///
/// EngineCore discovers generation and pooling capabilities from its workers.
/// Frontend-only tasks such as `render` are resolved above the engine client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EngineTask {
    /// A generation capability.
    Generation(GenerationTask),
    /// A pooling capability.
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
    use super::{EngineTask, GenerationTask, PoolingTask};

    #[test]
    fn engine_tasks_use_python_wire_literals() {
        let tasks = [
            EngineTask::Generation(GenerationTask::Generate),
            EngineTask::Generation(GenerationTask::Transcription),
            EngineTask::Generation(GenerationTask::Realtime),
            EngineTask::Pooling(PoolingTask::Embed),
            EngineTask::Pooling(PoolingTask::EmbedAndTokenClassify),
        ];

        let encoded = rmp_serde::to_vec(&tasks).unwrap();
        let literals: Vec<String> = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(
            literals,
            [
                "generate",
                "transcription",
                "realtime",
                "embed",
                "embed&token_classify",
            ]
        );

        let decoded: Vec<EngineTask> = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded, tasks);
    }

    #[test]
    fn unknown_engine_task_is_rejected() {
        let encoded = rmp_serde::to_vec(&["render"]).unwrap();
        assert!(rmp_serde::from_slice::<Vec<EngineTask>>(&encoded).is_err());
    }

    #[test]
    fn engine_tasks_decode_from_python_style_rmpv_strings() {
        let value = rmpv::Value::Array(vec!["generate".into(), "embed".into()]);
        assert_eq!(
            rmpv::ext::from_value::<Vec<EngineTask>>(value).unwrap(),
            vec![
                EngineTask::Generation(GenerationTask::Generate),
                EngineTask::Pooling(PoolingTask::Embed),
            ]
        );
    }
}
