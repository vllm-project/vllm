use vllm_engine_core_client::EngineCoreClient;

mod error;
mod output;
mod request;
mod request_metrics;

pub use error::{Error, Result};
pub use output::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStream,
    GenerateOutputStreamExt, GeneratePromptInfo,
};
pub use request::GenerateRequest;
pub use vllm_engine_core_client::protocol::{Logprobs, PositionLogprobs, TokenLogprob};

use crate::request_metrics::RequestMetricsTracker;

/// Thin generate-only facade over [`EngineCoreClient`].
///
/// This mirrors the narrow public shape of Python `AsyncLLM.generate()` and `abort()`, but
/// keeps the boundary close to raw engine-core requests and outputs.
pub struct Llm {
    client: EngineCoreClient,
    randomize_request_id: bool,
}

impl Llm {
    /// Create a new minimal LLM facade from an already connected engine-core client.
    pub fn new(client: EngineCoreClient) -> Self {
        Self {
            client,
            randomize_request_id: true,
        }
    }

    /// Control whether external request ids are randomized before reaching engine-core.
    pub fn with_request_id_randomization(mut self, enabled: bool) -> Self {
        self.randomize_request_id = enabled;
        self
    }

    /// Expose the underlying engine-core client for low-level utility/admin calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        &self.client
    }

    /// Submit one tokenized generate request and return a per-request output stream.
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        let prepared = req.prepare(self.randomize_request_id)?;
        let prompt_token_ids = prepared.prompt_token_ids().into();

        let request_metrics = RequestMetricsTracker::new(
            self.client.model_name().to_string(),
            prepared.engine_request.arrival_time,
            prepared.prompt_token_ids().len() as u32,
            (prepared.engine_request.sampling_params.as_ref()).map(|p| p.max_tokens),
            1,
        );
        let stream = self.client.call(prepared.engine_request).await?;

        Ok(GenerateOutputStream::new(
            prompt_token_ids,
            stream,
            request_metrics,
        ))
    }

    /// Shut down the underlying engine-core client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.client.shutdown().await?;
        Ok(())
    }
}
