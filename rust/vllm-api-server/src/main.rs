mod generated;
mod openai;

use openai::{ChatCompletionRequest, ChatMessage};

#[tokio::main]
async fn main() {
    // Verify OpenAI types work with serde
    let json = r#"{
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }"#;

    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "test");
    assert!(req.is_streaming());

    println!("OpenAI types working correctly");
}
