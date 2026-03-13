mod error;
mod output;
mod request;

pub use error::{Error, Result};
pub use output::GenerateOutput;
pub use request::GenerateRequest;
use vllm_engine_core_client::EngineCoreClient;

use crate::output::GenerateOutputStream;

pub struct Llm {
    client: EngineCoreClient,
}

impl Llm {
    pub fn new(client: EngineCoreClient) -> Self {
        Self { client }
    }

    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        let prepared = req.prepare()?;
        let output_kind = prepared.output_kind();
        let prompt_token_ids = prepared.prompt_token_ids().into();

        let stream = self.client.call(prepared.engine_request).await?;
        Ok(output::adapt_output_stream(
            output_kind,
            prompt_token_ids,
            stream,
        ))
    }

    pub async fn abort(&self, request_id: &str) -> Result<()> {
        self.client.abort(&[request_id.to_string()]).await?;
        Ok(())
    }

    pub async fn shutdown(self) -> Result<()> {
        self.client.shutdown().await?;
        Ok(())
    }
}
