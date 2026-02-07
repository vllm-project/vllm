use crate::openai::{ChatCompletionChunk, ChatChunkChoice, ChatDelta};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn create_chunk(
    id: &str,
    model: &str,
    content: Option<String>,
    role: Option<String>,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta { role, content },
            finish_reason,
        }],
    }
}

pub fn format_sse_data(chunk: &ChatCompletionChunk) -> String {
    format!("data: {}\n\n", serde_json::to_string(chunk).unwrap())
}

pub fn format_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}
