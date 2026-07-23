// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::{json, to_value};

use crate::{AssistantContentBlock, ChatContent, ChatContentPart, ChatMessage, ChatRole, Tool};

#[test]
fn chat_content_deserializes_from_raw_string() {
    let content: ChatContent = serde_json::from_value(json!("hello")).unwrap();
    assert_eq!(content, ChatContent::Text("hello".to_string()));
}

#[test]
fn chat_content_video_url_part_round_trips_through_serde() {
    let content = ChatContent::Parts(vec![ChatContentPart::VideoUrl {
        video_url: "https://example.com/demo.mp4".to_string(),
        uuid: Some("video-1".to_string()),
    }]);

    let value = to_value(&content).unwrap();
    assert_eq!(
        value,
        json!([{
            "type": "video_url",
            "video_url": "https://example.com/demo.mp4",
            "uuid": "video-1",
        }])
    );
    let decoded: ChatContent = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, content);
}

#[test]
fn chat_content_deserializes_from_openai_text_blocks() {
    let content: ChatContent =
        serde_json::from_value(json!([{ "type": "text", "text": "hello" }])).unwrap();
    assert_eq!(
        content,
        ChatContent::Parts(vec![ChatContentPart::text("hello")])
    );
}

#[test]
fn chat_content_from_string_like_values_builds_text() {
    assert_eq!(
        ChatContent::from("hello"),
        ChatContent::Text("hello".to_string())
    );
    assert_eq!(
        ChatContent::from("hello".to_string()),
        ChatContent::Text("hello".to_string())
    );
}

#[test]
fn chat_content_try_flattens_text_parts_without_separators() {
    let content = ChatContent::Parts(vec![
        ChatContentPart::text("hello"),
        ChatContentPart::text(" world"),
    ]);
    assert_eq!(content.try_flatten_to_text().unwrap(), "hello world");
}

#[test]
fn multimodal_content_parts_return_static_type_names() {
    let parts = [
        (ChatContentPart::image_url("image"), "image_url"),
        (ChatContentPart::video_url("video"), "video_url"),
        (ChatContentPart::input_audio("audio", None), "input_audio"),
        (ChatContentPart::audio_url("audio"), "audio_url"),
    ];

    for (part, expected) in parts {
        assert_eq!(part.as_text(), Err(expected));
        assert_eq!(
            ChatContent::Parts(vec![part]).try_flatten_to_text(),
            Err(expected)
        );
    }
}

#[test]
fn assistant_message_collects_visible_and_reasoning_text() {
    let message = ChatMessage::assistant_blocks(vec![
        AssistantContentBlock::Reasoning {
            text: "inner".to_string(),
        },
        AssistantContentBlock::Text {
            text: "outer".to_string(),
        },
    ]);

    assert_eq!(message.role(), ChatRole::Assistant);
    assert_eq!(message.text_content().unwrap(), "outer");
    assert_eq!(message.reasoning_content().as_deref(), Some("inner"));
}

#[test]
fn developer_message_round_trips_through_serde() {
    let message = ChatMessage::developer(
        "hello",
        Some(vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }),
            strict: Some(true),
        }]),
    );

    let value = to_value(&message).unwrap();
    let decoded: ChatMessage = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, message);
}
