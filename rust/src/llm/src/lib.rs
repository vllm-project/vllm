use vllm_engine_core_client::EngineCoreClient;

mod error;
mod output;
mod request;

pub use error::{Error, Result};
pub use output::{GenerateOutput, GenerateOutputStream};
pub use request::GenerateRequest;

/// Thin generate-only facade over [`EngineCoreClient`].
///
/// This mirrors the narrow public shape of Python `AsyncLLM.generate()` and `abort()`, but
/// keeps the boundary close to raw engine-core requests and outputs.
pub struct Llm {
    client: EngineCoreClient,
}

impl Llm {
    /// Create a new minimal LLM facade from an already connected engine-core client.
    pub fn new(client: EngineCoreClient) -> Self {
        Self { client }
    }

    /// Submit one tokenized generate request and return a per-request output stream.
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        let prepared = req.prepare()?;
        let output_kind = prepared.output_kind();
        let prompt_token_ids = prepared.prompt_token_ids().into();

        let stream = self.client.call(prepared.engine_request).await?;
        Ok(GenerateOutputStream::new(
            output_kind,
            prompt_token_ids,
            stream,
        ))
    }

    /// Abort one in-flight request by request ID.
    pub async fn abort(&self, request_id: &str) -> Result<()> {
        self.client.abort(&[request_id.to_string()]).await?;
        Ok(())
    }

    /// Shut down the underlying engine-core client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.client.shutdown().await?;
        Ok(())
    }
}
