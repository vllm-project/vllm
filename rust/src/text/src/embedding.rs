// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::mem::take;

use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_llm::{EncodeOutput, EncodeRequest, PoolingParams as LlmPoolingParams, PoolingTask};

use crate::error::{Error, Result};
use crate::{Prompt, PromptTruncation, TextLlm, TextRequestProcessor};

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("dimensions must be greater than zero")]
    InvalidDimensions,
    #[error(
        "embedding request `{request_id}` expected a rank-1 pooling tensor, got shape {shape:?}"
    )]
    OutputRank {
        request_id: String,
        shape: Vec<usize>,
    },
}

impl EmbeddingError {
    pub(crate) fn is_request_validation_error(&self) -> bool {
        !matches!(self, Self::OutputRank { .. })
    }
}

/// User-facing parameters for one embedding operation.
///
/// Original Python definitions:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/pooling_params.py>
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/config/pooler.py>
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingParams {
    /// Reduce the dimensions of embeddings when the model supports
    /// Matryoshka representation.
    /// `None` lets engine-core resolve the model's default.
    pub dimensions: Option<u32>,
    /// Whether to apply activation function to the pooler outputs.
    /// `None` uses the pooler's default, which is `true` in most cases.
    pub use_activation: Option<bool>,
}

/// One prompt-level embedding request accepted by [`crate::TextLlm`].
///
/// Original Python request fields:
/// <https://github.com/vllm-project/vllm/blob/6ec92bcbc8/vllm/entrypoints/pooling/base/protocol.py>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Request ID used throughout inference and returned in the response.
    pub request_id: String,
    /// Prompt text or pre-tokenized prompt IDs.
    pub prompt: Prompt,
    /// Embedding parameters forwarded to engine-core for model-aware resolution.
    pub params: EmbeddingParams,
    /// Optional typed prompt-truncation policy.
    pub prompt_truncation: Option<PromptTruncation>,
    /// Whether to add special tokens, such as BOS, to the prompt during
    /// tokenization.
    pub add_special_tokens: bool,
    /// Request scheduling priority. Lower values are handled earlier. A
    /// nonzero priority requires priority scheduling in the served model.
    pub priority: i32,
    /// Random salt used to isolate prefix-cache entries across users. The
    /// value should remain protected from third parties and carry enough
    /// entropy to prevent prompt guessing.
    pub cache_salt: Option<String>,
    /// Optional tracing headers forwarded to engine-core.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// Override data-parallel rank routing.
    pub data_parallel_rank: Option<u32>,
    /// Stable session identity shared by related requests.
    pub session_id: Option<String>,
    /// LoRA adapter selected for this request.
    pub lora_request: Option<LoraRequest>,
    /// Wall-clock unix timestamp when this request arrived at the frontend.
    pub arrival_time: Option<f64>,
}

impl EmbeddingRequest {
    /// Return one minimal request fixture for tests.
    pub fn for_test() -> Self {
        Self {
            request_id: "test-embedding".to_string(),
            prompt: Prompt::Text("test".to_string()),
            params: EmbeddingParams::default(),
            prompt_truncation: None,
            add_special_tokens: true,
            priority: 0,
            cache_salt: None,
            trace_headers: None,
            data_parallel_rank: None,
            session_id: None,
            lora_request: None,
            arrival_time: None,
        }
    }

    fn validate(&self) -> Result<()> {
        if matches!(&self.prompt, Prompt::TokenIds(token_ids) if token_ids.is_empty()) {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id.clone(),
            });
        }
        if self.params.dimensions == Some(0) {
            return Err(EmbeddingError::InvalidDimensions.into());
        }
        Ok(())
    }
}

/// Final embedding result for one prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingOutput {
    /// Stable caller-supplied request ID.
    pub request_id: String,
    /// Token IDs submitted to the pooling model after truncation.
    pub prompt_token_ids: Vec<u32>,
    /// Rank-1 embedding vector.
    pub embedding: Vec<f32>,
    /// Number of prompt tokens served from cache.
    pub cached_token_count: usize,
}

impl EmbeddingOutput {
    fn from_encode(request_id: String, output: EncodeOutput) -> Result<Self> {
        if output.output.shape.len() != 1 {
            return Err(EmbeddingError::OutputRank {
                request_id,
                shape: output.output.shape,
            }
            .into());
        }

        Ok(Self {
            request_id,
            prompt_token_ids: output.prompt_token_ids,
            embedding: output.output.data,
            cached_token_count: output.cached_token_count,
        })
    }
}

impl TextRequestProcessor {
    /// Tokenize, validate, and lower one prompt-level embedding request.
    pub fn prepare_embedding(&self, mut request: EmbeddingRequest) -> Result<EncodeRequest> {
        request.validate()?;

        if request.arrival_time.is_none() {
            request.arrival_time = Some(vllm_llm::current_unix_timestamp_secs());
        }

        let prompt_token_ids = self.prepare_prompt_tokens(
            take(&mut request.prompt),
            request.add_special_tokens,
            request.prompt_truncation,
            None,
        )?;
        self.validate_prompt_tokens(&request.request_id, &prompt_token_ids)?;

        Ok(EncodeRequest {
            request_id: request.request_id,
            prompt_token_ids,
            task: PoolingTask::Embed,
            pooling_params: LlmPoolingParams {
                use_activation: request.params.use_activation,
                dimensions: request.params.dimensions,
                step_tag_id: None,
                returned_token_ids: None,
            },
            arrival_time: request.arrival_time,
            cache_salt: request.cache_salt,
            trace_headers: request.trace_headers,
            priority: request.priority,
            data_parallel_rank: request.data_parallel_rank,
            session_id: request.session_id,
            lora_request: request.lora_request,
        })
    }
}

impl TextLlm {
    /// Embed one text or token-ID prompt through the model's sequence pooler.
    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingOutput> {
        let encode_request = self.processor.prepare_embedding(request)?;
        let request_id = encode_request.request_id.clone();
        let output = self.llm.encode(encode_request).await?;
        EmbeddingOutput::from_encode(request_id, output)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use vllm_llm::PoolingOutput;
    use vllm_tokenizer::DynTokenizer;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;
    use crate::TextBackend;

    struct FakeEmbeddingBackend {
        tokenizer: DynTokenizer,
    }

    impl TextBackend for FakeEmbeddingBackend {
        fn tokenizer(&self) -> DynTokenizer {
            self.tokenizer.clone()
        }

        fn model_id(&self) -> &str {
            "test-embedding-model"
        }

        fn model_vocab_size(&self) -> usize {
            512
        }

        fn tokenizer_vocab_size(&self) -> usize {
            512
        }
    }

    fn processor() -> TextRequestProcessor {
        let tokenizer = TestTokenizer::new().with_bos_token("<s>", 300).with_vocab_size(512);
        TextRequestProcessor::new(
            Arc::new(FakeEmbeddingBackend {
                tokenizer: Arc::new(tokenizer),
            }),
            4,
        )
    }

    fn request(prompt: Prompt) -> EmbeddingRequest {
        EmbeddingRequest {
            prompt,
            arrival_time: Some(42.5),
            ..EmbeddingRequest::for_test()
        }
    }

    #[test]
    fn preparation_tokenizes_and_forwards_unset_pooling_params() {
        let prepared =
            processor().prepare_embedding(request(Prompt::Text("abc".to_string()))).unwrap();

        expect![[r#"
            EncodeRequest {
                request_id: "test-embedding",
                prompt_token_ids: [
                    300,
                    97,
                    98,
                    99,
                ],
                task: Embed,
                pooling_params: PoolingParams {
                    use_activation: None,
                    dimensions: None,
                    step_tag_id: None,
                    returned_token_ids: None,
                },
                arrival_time: Some(
                    42.5,
                ),
                cache_salt: None,
                trace_headers: None,
                priority: 0,
                data_parallel_rank: None,
                session_id: None,
                lora_request: None,
            }
        "#]]
        .assert_debug_eq(&prepared);
    }

    #[test]
    fn preparation_forwards_explicit_pooling_params() {
        let mut embedding_request = request(Prompt::TokenIds(vec![1, 2]));
        embedding_request.params = EmbeddingParams {
            dimensions: Some(4),
            use_activation: Some(true),
        };

        let prepared = processor().prepare_embedding(embedding_request).unwrap();

        assert_eq!(prepared.pooling_params.use_activation, Some(true));
        assert_eq!(prepared.pooling_params.dimensions, Some(4));
    }

    #[test]
    fn preparation_applies_typed_truncation() {
        let processor = processor();
        let mut right = request(Prompt::TokenIds(vec![1, 2, 3, 4, 5]));
        right.prompt_truncation = Some(PromptTruncation {
            limit: crate::PromptTruncationLimit::Fixed(3),
            side: crate::TruncationSide::Right,
        });
        let mut left = right.clone();
        left.prompt_truncation.as_mut().unwrap().side = crate::TruncationSide::Left;

        let right_ids = processor.prepare_embedding(right).unwrap().prompt_token_ids;
        let left_ids = processor.prepare_embedding(left).unwrap().prompt_token_ids;

        expect![[r#"
            (
                [
                    1,
                    2,
                    3,
                ],
                [
                    3,
                    4,
                    5,
                ],
            )
        "#]]
        .assert_debug_eq(&(right_ids, left_ids));
    }

    #[test]
    fn preparation_rejects_out_of_vocabulary_prompt_ids() {
        let error = processor()
            .prepare_embedding(request(Prompt::TokenIds(vec![1, 512])))
            .unwrap_err();

        assert!(matches!(
            error,
            Error::TokenIds(crate::TokenIdsError::OutOfVocab {
                parameter: "prompt",
                token_ids,
                vocab_size: 512,
            }) if token_ids == vec![512]
        ));
    }

    #[test]
    fn preparation_enforces_embedding_length_limits() {
        let processor = processor();
        let too_long = processor
            .prepare_embedding(request(Prompt::TokenIds(vec![1, 2, 3, 4, 5])))
            .unwrap_err();
        let mut invalid_truncation = request(Prompt::TokenIds(vec![1, 2, 3]));
        invalid_truncation.prompt_truncation = Some(PromptTruncation {
            limit: crate::PromptTruncationLimit::Fixed(5),
            side: crate::TruncationSide::Right,
        });
        let invalid_truncation = processor.prepare_embedding(invalid_truncation).unwrap_err();

        expect![[r#"
            [
                "this model's maximum context length is 4 tokens, but the prompt contains 5 input tokens",
                "truncate_prompt_tokens=5 exceeds the available input budget of 4 tokens",
            ]
        "#]]
        .assert_debug_eq(&vec![too_long.to_string(), invalid_truncation.to_string()]);
    }

    #[test]
    fn preparation_rejects_zero_dimensions() {
        let mut embedding_request = request(Prompt::TokenIds(vec![1, 2]));
        embedding_request.params.dimensions = Some(0);

        let error = processor().prepare_embedding(embedding_request).unwrap_err();

        assert!(matches!(
            error,
            Error::Embedding(EmbeddingError::InvalidDimensions)
        ));
    }

    #[test]
    fn embedding_output_requires_a_rank_one_tensor_and_preserves_external_id() {
        let output = EmbeddingOutput::from_encode(
            "external-id".to_string(),
            EncodeOutput {
                request_id: "internal-id".to_string(),
                prompt_token_ids: vec![1, 2],
                output: PoolingOutput {
                    shape: vec![2],
                    data: vec![0.25, -0.5],
                },
                cached_token_count: 1,
            },
        )
        .unwrap();

        expect![[r#"
            EmbeddingOutput {
                request_id: "external-id",
                prompt_token_ids: [
                    1,
                    2,
                ],
                embedding: [
                    0.25,
                    -0.5,
                ],
                cached_token_count: 1,
            }
        "#]]
        .assert_debug_eq(&output);

        let error = EmbeddingOutput::from_encode(
            "external-id".to_string(),
            EncodeOutput {
                request_id: "internal-id".to_string(),
                prompt_token_ids: vec![1, 2],
                output: PoolingOutput {
                    shape: vec![1, 2],
                    data: vec![0.25, -0.5],
                },
                cached_token_count: 0,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Embedding(EmbeddingError::OutputRank { request_id, shape })
                if request_id == "external-id" && shape == vec![1, 2]
        ));
    }
}
