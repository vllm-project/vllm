// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `POST /v1/decisions` request and response bodies.
//!
//! Original Python definition: `vllm/entrypoints/openai/decisions/protocol.py`.

use std::collections::HashSet;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::error::{ApiError, invalid_request};

/// `ShortText` of the Python protocol, in characters.
const MAX_SHORT_TEXT: usize = 1_048_576;
/// `InputText` of the Python protocol, in characters.
const MAX_INPUT_TEXT: usize = 10_485_760;
const MAX_INPUT_MESSAGES: usize = 131_072;
const MAX_INPUT_PARTS: usize = 16_384;
const MAX_QUESTIONS: usize = 200;

/// `POST /v1/decisions` request body.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub model: String,
    /// The text the questions are about.
    pub input: DecisionInput,
    pub questions: Vec<DecisionQuestion>,
    pub safety_identifier: Option<String>,
    /// vLLM extension: joint reads, diffusion models only.
    #[serde(default)]
    pub joint: bool,
    /// Salt for prefix cache isolation, applied to every read.
    pub cache_salt: Option<String>,
}

impl DecisionRequest {
    /// The limits that Python's protocol declares on its fields.
    pub fn check_limits(&self) -> Result<(), ApiError> {
        check_length("model", &self.model, MAX_SHORT_TEXT)?;
        if self.questions.is_empty() || self.questions.len() > MAX_QUESTIONS {
            return Err(invalid_request!(
                param = "questions",
                "questions: must have between 1 and {MAX_QUESTIONS} items"
            ));
        }
        for question in &self.questions {
            question.check_limits()?;
        }
        if let Some(safety_identifier) = &self.safety_identifier {
            check_length("safety_identifier", safety_identifier, 128)?;
        }
        if let Some(cache_salt) = &self.cache_salt
            && cache_salt.chars().count() > 1024
        {
            return Err(invalid_request!(
                param = "cache_salt",
                "cache_salt: must have at most 1024 characters"
            ));
        }
        Ok(())
    }
}

fn check_length(field: &'static str, text: &str, max: usize) -> Result<(), ApiError> {
    if text.chars().count() > max {
        return Err(invalid_request!(
            param = field,
            "{field}: must have at most {max} characters"
        ));
    }
    Ok(())
}

/// The request's `input`, joined into the text the questions are about: a
/// string, or the text of user messages, one message per paragraph.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "Value")]
pub struct DecisionInput(pub String);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputMessage {
    #[expect(dead_code, reason = "only `user` deserializes")]
    role: UserRole,
    content: InputContent,
    #[serde(rename = "type", default)]
    #[expect(dead_code, reason = "only `message` deserializes")]
    kind: MessageKind,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum UserRole {
    User,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MessageKind {
    #[default]
    Message,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum InputContent {
    Text(String),
    Parts(Vec<InputText>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputText {
    #[serde(rename = "type")]
    #[expect(dead_code, reason = "only `input_text` deserializes")]
    kind: InputTextKind,
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputTextKind {
    InputText,
}

impl TryFrom<Value> for DecisionInput {
    type Error = String;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let too_long = |text: &str| text.chars().count() > MAX_INPUT_TEXT;
        let messages = match value {
            Value::String(text) if too_long(&text) => {
                return Err(format!("must have at most {MAX_INPUT_TEXT} characters"));
            }
            Value::String(text) => return Ok(Self(text)),
            Value::Array(messages) => messages,
            _ => return Err("must be a string or a list of messages".to_owned()),
        };
        if messages.len() > MAX_INPUT_MESSAGES {
            return Err(format!("must have at most {MAX_INPUT_MESSAGES} messages"));
        }
        // Like Python, an image part is refused before the rest of the
        // message is checked.
        let has_image = |message: &Value| {
            (message.get("content").and_then(Value::as_array)).is_some_and(|parts| {
                (parts.iter())
                    .any(|part| part.get("type").and_then(Value::as_str) == Some("input_image"))
            })
        };
        if messages.iter().any(has_image) {
            return Err("Decisions supports text input only; images are not supported".to_owned());
        }
        let messages = (messages.into_iter())
            .map(serde_json::from_value::<InputMessage>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let mut paragraphs = Vec::with_capacity(messages.len());
        for message in messages {
            paragraphs.push(match message.content {
                InputContent::Text(text) if too_long(&text) => {
                    return Err(format!("must have at most {MAX_INPUT_TEXT} characters"));
                }
                InputContent::Text(text) => text,
                InputContent::Parts(parts) if parts.len() > MAX_INPUT_PARTS => {
                    return Err(format!("must have at most {MAX_INPUT_PARTS} content parts"));
                }
                InputContent::Parts(parts) => {
                    if parts.iter().any(|part| too_long(&part.text)) {
                        return Err(format!("must have at most {MAX_INPUT_TEXT} characters"));
                    }
                    let texts: Vec<_> = parts.into_iter().map(|part| part.text).collect();
                    texts.join("\n")
                }
            });
        }
        Ok(Self(paragraphs.join("\n\n")))
    }
}

/// One question of a decision.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionQuestion {
    /// A yes-or-no question; the answer is the probability of yes.
    Predicate {
        instructions: String,
        #[serde(default, deserialize_with = "non_null_name")]
        name: Option<String>,
    },
    /// Pick one of the choices.
    Choice {
        instructions: String,
        #[serde(default, deserialize_with = "non_null_name")]
        name: Option<String>,
        choices: Vec<ChoiceOption>,
    },
    /// Rate on the ordered levels.
    Score {
        instructions: String,
        #[serde(default, deserialize_with = "non_null_name")]
        name: Option<String>,
        levels: Vec<ScoreLevel>,
    },
}

/// A `name` may be left out, but not set to null.
fn non_null_name<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    String::deserialize(deserializer)
        .map(Some)
        .map_err(|_| D::Error::custom("name must be a string when provided"))
}

impl DecisionQuestion {
    fn check_limits(&self) -> Result<(), ApiError> {
        let (instructions, name) = match self {
            Self::Predicate { instructions, name }
            | Self::Choice {
                instructions, name, ..
            }
            | Self::Score {
                instructions, name, ..
            } => (instructions, name),
        };
        check_length("instructions", instructions, MAX_SHORT_TEXT)?;
        if let Some(name) = name {
            check_length("name", name, MAX_SHORT_TEXT)?;
        }
        match self {
            Self::Predicate { .. } => {}
            Self::Choice { choices, .. } => {
                if choices.len() < 2 || choices.len() > 255 {
                    return Err(invalid_request!(
                        param = "choices",
                        "choices: must have between 2 and 255 items"
                    ));
                }
                let mut seen = HashSet::new();
                for choice in choices {
                    check_length("description", &choice.description, MAX_SHORT_TEXT)?;
                    if let ChoiceValue::Text(value) = &choice.value {
                        check_length("value", value, MAX_SHORT_TEXT)?;
                    }
                    if !seen.insert(&choice.value) {
                        return Err(invalid_request!(
                            param = "choices",
                            "choice values must be distinct"
                        ));
                    }
                }
            }
            Self::Score { levels, .. } => {
                if levels.len() < 2 || levels.len() > 10 {
                    return Err(invalid_request!(
                        param = "levels",
                        "levels: must have between 2 and 10 items"
                    ));
                }
                for level in levels {
                    check_length("label", &level.label, MAX_SHORT_TEXT)?;
                    check_length("description", &level.description, MAX_SHORT_TEXT)?;
                }
            }
        }
        Ok(())
    }
}

/// A choice's value: a string or a boolean, returned as sent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChoiceValue {
    Bool(bool),
    Text(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceOption {
    pub value: ChoiceValue,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreLevel {
    pub label: String,
    #[serde(default)]
    pub description: String,
}

/// `POST /v1/decisions` response body. Unset names serialize as `null`, like
/// Python's.
#[derive(Debug, Clone, Serialize)]
pub struct DecisionResponse {
    pub model: String,
    /// One answer per question, in request order.
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Predicate {
        name: Option<String>,
        probability: f64,
    },
    Choice {
        name: Option<String>,
        choice: ChoiceValue,
        probabilities: Vec<ChoiceProbability>,
        /// The chosen label's probability over the whole vocabulary.
        confidence: f64,
    },
    Score {
        name: Option<String>,
        /// The expected level, 0-indexed.
        score: f64,
        probabilities: Vec<ScoreProbability>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ChoiceProbability {
    pub value: ChoiceValue,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreProbability {
    pub value: usize,
    pub label: String,
    pub probability: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: usize,
    pub cache_write_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: usize,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn input(value: Value) -> Result<String, String> {
        DecisionInput::try_from(value).map(|input| input.0)
    }

    #[test]
    fn input_joins_messages_like_python() {
        assert_eq!(input(json!("text")).unwrap(), "text");
        let messages = json!([
            {"role": "user", "content": "first"},
            {"role": "user", "type": "message", "content": [
                {"type": "input_text", "text": "second"},
                {"type": "input_text", "text": "third"},
            ]},
        ]);
        assert_eq!(input(messages).unwrap(), "first\n\nsecond\nthird");
    }

    #[test]
    fn input_refuses_images_and_other_roles() {
        let image = json!([{"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/a.png"},
        ]}]);
        assert_eq!(
            input(image).unwrap_err(),
            "Decisions supports text input only; images are not supported"
        );
        assert!(input(json!([{"role": "assistant", "content": "x"}])).is_err());
        assert!(input(json!([{"role": "user", "content": "x", "extra": 1}])).is_err());
    }

    #[test]
    fn questions_follow_the_python_protocol() {
        let parse = |value: Value| serde_json::from_value::<DecisionQuestion>(value);
        let question = parse(json!({
            "type": "choice", "instructions": "Pick", "choices": [{"value": true}, {"value": "x"}],
        }))
        .unwrap();
        assert!(matches!(
            question,
            DecisionQuestion::Choice { name: None, .. }
        ));
        question.check_limits().unwrap();

        let error = parse(json!({"type": "predicate", "instructions": "Q", "name": null}));
        assert!(error.unwrap_err().to_string().contains("name must be a string when provided"));
        assert!(parse(json!({"type": "predicate", "instructions": "Q", "extra": 1})).is_err());
        assert!(parse(json!({"type": "predicate", "instructions": 1})).is_err());

        // A string and a boolean with the same text are distinct values.
        let distinct = parse(json!({
            "type": "choice", "instructions": "Pick", "choices": [{"value": true}, {"value": "true"}],
        }))
        .unwrap();
        distinct.check_limits().unwrap();
        let repeated = parse(json!({
            "type": "choice", "instructions": "Pick", "choices": [{"value": "a"}, {"value": "a"}],
        }))
        .unwrap();
        let error = repeated.check_limits().unwrap_err();
        assert!(format!("{error:?}").contains("choice values must be distinct"));
    }
}
