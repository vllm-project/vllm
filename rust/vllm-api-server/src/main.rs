mod generated;

use generated::vllm_engine_client::VllmEngineClient;

#[tokio::main]
async fn main() {
    println!("vllm-api-server - proto types available");

    // Verify types compile
    let _: Option<generated::GenerateRequest> = None;
    let _: Option<generated::SamplingParams> = None;
}
