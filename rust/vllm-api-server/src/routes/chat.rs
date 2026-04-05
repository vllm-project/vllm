use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::generated::generate_response::Response as GrpcResponse;
use crate::grpc::VllmClient;
use crate::openai::{
    create_chunk, format_sse_data, format_sse_done,
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatMessage, Usage,
};
use crate::server::AppState;

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.is_streaming() {
        match handle_streaming(state, request).await {
            Ok(response) => response,
            Err(e) => {
                tracing::error!("Streaming chat completion failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    } else {
        match handle_non_streaming(state, request).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(e) => {
                tracing::error!("Chat completion failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                )
                    .into_response()
            }
        }
    }
}

async fn handle_streaming(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<Response, Box<dyn std::error::Error + Send + Sync>> {
    let request_id = Uuid::new_v4().to_string();
    let chat_id = format!("chatcmpl-{}", request_id);
    let model_name = state.model_name.clone();

    // Apply chat template and tokenize
    let prompt = apply_chat_template(&request.messages, state.chat_template.as_deref());
    let encoding = state
        .tokenizer
        .encode(prompt.clone(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    tracing::debug!(
        "Streaming request {}: {} prompt tokens",
        request_id,
        token_ids.len()
    );

    // Build gRPC request
    let grpc_request = VllmClient::build_generate_request(
        request_id,
        token_ids,
        prompt,
        &request,
        true, // streaming
    );

    // Call gRPC server
    let mut client = state.grpc_client.lock().await;
    let grpc_stream = client.generate(grpc_request).await?;

    // Create SSE stream
    let tokenizer = state.tokenizer.clone();
    let stream = async_stream::stream! {
        let mut grpc_stream = grpc_stream;
        let mut first_chunk = true;

        while let Some(result) = grpc_stream.next().await {
            match result {
                Ok(response) => {
                    if let Some(grpc_response) = response.response {
                        match grpc_response {
                            GrpcResponse::Chunk(chunk) => {
                                if !chunk.token_ids.is_empty() {
                                    let text = tokenizer
                                        .decode(&chunk.token_ids, true)
                                        .unwrap_or_default();

                                    let sse_chunk = if first_chunk {
                                        first_chunk = false;
                                        create_chunk(
                                            &chat_id,
                                            &model_name,
                                            Some(text),
                                            Some("assistant".to_string()),
                                            None,
                                        )
                                    } else {
                                        create_chunk(
                                            &chat_id,
                                            &model_name,
                                            Some(text),
                                            None,
                                            None,
                                        )
                                    };
                                    yield Ok::<_, std::io::Error>(format_sse_data(&sse_chunk));
                                }
                            }
                            GrpcResponse::Complete(complete) => {
                                // Send final chunk with finish_reason
                                let final_chunk = create_chunk(
                                    &chat_id,
                                    &model_name,
                                    None,
                                    None,
                                    Some(complete.finish_reason),
                                );
                                yield Ok(format_sse_data(&final_chunk));
                                yield Ok(format_sse_done());
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("gRPC stream error: {}", e);
                    break;
                }
            }
        }
    };

    let body = Body::from_stream(stream);

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap())
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
    let request_id = Uuid::new_v4().to_string();

    // Apply chat template and tokenize
    let prompt = apply_chat_template(&request.messages, state.chat_template.as_deref());
    let encoding = state
        .tokenizer
        .encode(prompt.clone(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_tokens = token_ids.len() as u32;

    tracing::debug!(
        "Request {}: {} prompt tokens",
        request_id,
        prompt_tokens
    );

    // Build gRPC request
    let grpc_request = VllmClient::build_generate_request(
        request_id.clone(),
        token_ids,
        prompt,
        &request,
        false,
    );

    // Call gRPC server
    let mut client = state.grpc_client.lock().await;
    let mut stream = client.generate(grpc_request).await?;

    // Collect response
    let mut output_tokens: Vec<u32> = Vec::new();
    let mut finish_reason = String::new();
    let mut completion_tokens = 0u32;

    while let Some(response) = stream.message().await? {
        if let Some(grpc_response) = response.response {
            match grpc_response {
                GrpcResponse::Chunk(chunk) => {
                    output_tokens.extend(chunk.token_ids);
                }
                GrpcResponse::Complete(complete) => {
                    if !complete.output_ids.is_empty() {
                        output_tokens = complete.output_ids;
                    }
                    finish_reason = complete.finish_reason;
                    completion_tokens = complete.completion_tokens;
                    break;
                }
            }
        }
    }

    // Detokenize output
    let output_text = state
        .tokenizer
        .decode(&output_tokens, true)
        .map_err(|e| format!("Detokenization failed: {}", e))?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", request_id),
        object: "chat.completion".to_string(),
        created,
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: output_text,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn apply_chat_template(messages: &[ChatMessage], _template: Option<&str>) -> String {
    // TODO: Add minijinja for proper Jinja2 template support
    // For now, use a simple fallback template
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("<|{}|>\n{}\n", msg.role, msg.content));
            }
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}
