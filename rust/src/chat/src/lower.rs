use vllm_engine_core_client::protocol::RequestOutputKind;
use vllm_llm::GenerateRequest;

use crate::error::Result;
use crate::renderer::RenderedPrompt;
use crate::request::ChatRequest;
use crate::tokenizer::Tokenizer;

#[derive(Debug)]
pub(crate) struct PreparedChatRequest {
    pub request_id: String,
    pub generate_request: GenerateRequest,
}

pub(crate) fn lower_chat_request(
    request: ChatRequest,
    rendered: RenderedPrompt,
    tokenizer: &dyn Tokenizer,
) -> Result<PreparedChatRequest> {
    let ChatRequest {
        request_id,
        messages: _,
        mut sampling_params,
        chat_options: _,
    } = request;

    let (prompt_token_ids, _prompt_text) = match rendered {
        RenderedPrompt::Text { prompt } => (tokenizer.encode(&prompt, false)?, Some(prompt)),
        RenderedPrompt::Tokens {
            prompt_token_ids,
            prompt_text,
        } => (prompt_token_ids, prompt_text),
    };

    sampling_params.output_kind = RequestOutputKind::Cumulative;

    let generate_request = GenerateRequest {
        request_id: request_id.clone(),
        prompt_token_ids,
        sampling_params,
        arrival_time: None,
        cache_salt: None,
        trace_headers: None,
        priority: 0,
        data_parallel_rank: None,
        reasoning_ended: None,
        lora_request: None,
    };

    Ok(PreparedChatRequest {
        request_id,
        generate_request,
    })
}
