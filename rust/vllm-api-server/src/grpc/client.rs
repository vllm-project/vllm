use tonic::transport::Channel;
use crate::generated::{
    vllm_engine_client::VllmEngineClient,
    GenerateRequest, GenerateResponse, SamplingParams,
    TokenizedInput, HealthCheckRequest,
    generate_request::Input,
};
use crate::openai::ChatCompletionRequest;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GrpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(#[from] tonic::transport::Error),
    #[error("RPC failed: {0}")]
    RpcFailed(#[from] tonic::Status),
}

#[derive(Clone)]
pub struct VllmClient {
    client: VllmEngineClient<Channel>,
}

impl VllmClient {
    pub async fn connect(addr: &str) -> Result<Self, GrpcError> {
        let client = VllmEngineClient::connect(format!("http://{}", addr)).await?;
        Ok(Self { client })
    }

    pub async fn health_check(&mut self) -> Result<bool, GrpcError> {
        let response = self.client.health_check(HealthCheckRequest {}).await?;
        Ok(response.into_inner().healthy)
    }

    pub fn build_generate_request(
        request_id: String,
        token_ids: Vec<u32>,
        original_text: String,
        req: &ChatCompletionRequest,
        stream: bool,
    ) -> GenerateRequest {
        GenerateRequest {
            request_id,
            input: Some(Input::Tokenized(TokenizedInput {
                original_text,
                input_ids: token_ids,
            })),
            sampling_params: Some(Self::build_sampling_params(req)),
            stream,
        }
    }

    fn build_sampling_params(req: &ChatCompletionRequest) -> SamplingParams {
        SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p.unwrap_or(1.0),
            max_tokens: req.max_tokens,
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            presence_penalty: req.presence_penalty.unwrap_or(0.0),
            seed: req.seed.map(|s| s as i32),
            stop: req.stop.clone().unwrap_or_default(),
            // Defaults for other fields
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            min_tokens: 0,
            stop_token_ids: vec![],
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ignore_eos: false,
            n: 1,
            logprobs: None,
            prompt_logprobs: None,
            include_stop_str_in_output: false,
            logit_bias: std::collections::HashMap::new(),
            truncate_prompt_tokens: None,
            constraint: None,
        }
    }

    pub async fn generate(
        &mut self,
        request: GenerateRequest,
    ) -> Result<tonic::Streaming<GenerateResponse>, GrpcError> {
        let response = self.client.generate(request).await?;
        Ok(response.into_inner())
    }
}
