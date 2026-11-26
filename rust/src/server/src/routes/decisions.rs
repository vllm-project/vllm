// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Decisions API (`POST /v1/decisions`): the probability of each answer to
//! questions about a text, read from the next-token logprobs of the option
//! labels.
//!
//! Only the next-token read strategy is implemented. The Python launcher
//! passes the strategy Python's `select_read_strategy` picks for the model
//! ([`crate::DecisionsReadStrategy`]); without one, this answers 501.

mod next_token;
mod questions;
mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
pub(crate) use next_token::NextTokenReads;
use tracing_futures::Instrument as _;

use self::next_token::{MAX_QUESTIONS, QuestionRead, ReadOptions};
use self::questions::{Question, QuestionOption, QuestionType, argmax};
use self::types::{
    ChoiceProbability, DecisionAnswer, DecisionQuestion, DecisionRequest, DecisionResponse,
    DecisionUsage, ScoreProbability,
};
use crate::error::{ApiError, bail_invalid_request};
use crate::state::AppState;
use crate::utils::resolve_request_context;

/// Answer every question of one request, each as its own engine request.
pub async fn decisions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Result<Json<DecisionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request?;
    let reads = (state.decisions()).expect("the route is registered only with a read strategy");
    let ctx = resolve_request_context(&headers, None)?;
    let lora_resolution = state.resolve_model_with_loras(Some(&request.model)).await;
    lora_resolution.model_names.check(Some(&request.model))?;
    request.check_limits()?;
    if request.questions.len() > MAX_QUESTIONS {
        bail_invalid_request!("questions: at most {MAX_QUESTIONS} for this model");
    }
    let questions = (request.questions.iter()).map(read_question).collect::<Result<Vec<_>, _>>()?;
    if request.joint {
        bail_invalid_request!(
            "joint: this model reads each question on its own; joint reads need DiffusionGemma"
        );
    }

    let request_id = format!("decision-{}", ctx.request_id);
    let span = tracing::info_span!("decisions", request_id = %request_id);
    let options = ReadOptions {
        request_id,
        chat_template_kwargs: None,
        lora_request: lora_resolution.lora_request.clone(),
        // Like Python, a decision runs at the default priority on any rank.
        priority: 0,
        cache_salt: request.cache_salt.clone(),
        data_parallel_rank: None,
        session_id: None,
    };
    let reads = (reads.read(&state.chat, &questions, None, &request.input.0, options))
        .instrument(span)
        .await?;

    let mut usage = DecisionUsage::default();
    for read in &reads {
        usage.input_tokens += read.input_tokens;
        usage.output_tokens += read.output_tokens;
        usage.input_tokens_details.cached_tokens += read.cached_tokens;
        usage.input_tokens_details.cache_write_tokens += read.cache_write_tokens;
    }
    usage.total_tokens = usage.input_tokens + usage.output_tokens;
    let answers = (request.questions.into_iter())
        .zip(&reads)
        .map(|(question, read)| answer(question, read))
        .collect();
    Ok(Json(DecisionResponse {
        model: lora_resolution.response_model(),
        answers,
        usage,
    })
    .into_response())
}

/// The read of one question: a predicate as yes or no, a choice or a score as
/// options labeled with letters. Python's `make_read_question`.
fn read_question(question: &DecisionQuestion) -> Result<Question, ApiError> {
    let (question_type, instructions, options) = match question {
        DecisionQuestion::Predicate { instructions, .. } => {
            let side = |name: &str| QuestionOption {
                name: name.to_owned(),
                description: None,
            };
            (
                QuestionType::Noul,
                instructions,
                vec![side("yes"), side("no")],
            )
        }
        DecisionQuestion::Choice {
            instructions,
            choices,
            ..
        } => {
            let options = (choices.iter())
                .map(|choice| QuestionOption {
                    // Like Python's `json.dumps(value, ensure_ascii=False)`.
                    name: serde_json::to_string(&choice.value).expect("a string or a boolean"),
                    description: Some(choice.description.clone()),
                })
                .collect();
            (QuestionType::Choice, instructions, options)
        }
        DecisionQuestion::Score {
            instructions,
            levels,
            ..
        } => {
            let options = (levels.iter())
                .map(|level| QuestionOption {
                    name: level.label.clone(),
                    description: Some(level.description.clone()),
                })
                .collect();
            (QuestionType::Choice, instructions, options)
        }
    };
    let limit = question_type.label_set().len();
    if options.len() > limit {
        bail_invalid_request!("This model supports at most {limit} choices per question");
    }
    Ok(Question {
        question_type,
        instructions: instructions.clone(),
        options,
    })
}

/// Python's `make_answer`.
fn answer(question: DecisionQuestion, read: &QuestionRead) -> DecisionAnswer {
    let probs = &read.probs;
    let top = argmax(probs);
    // The chosen label's probability over the whole vocabulary, so a read
    // that mostly wanted a non-label token reports low confidence.
    let confidence = probs[top] * read.label_mass;
    match question {
        DecisionQuestion::Predicate { name, .. } => DecisionAnswer::Predicate {
            name,
            probability: probs[0],
        },
        DecisionQuestion::Choice { name, choices, .. } => DecisionAnswer::Choice {
            name,
            choice: choices[top].value.clone(),
            probabilities: (choices.into_iter())
                .zip(probs)
                .map(|(choice, &probability)| ChoiceProbability {
                    value: choice.value,
                    probability,
                })
                .collect(),
            confidence,
        },
        DecisionQuestion::Score { name, levels, .. } => DecisionAnswer::Score {
            name,
            score: probs.iter().enumerate().map(|(value, p)| value as f64 * p).sum(),
            probabilities: (levels.into_iter())
                .zip(probs)
                .enumerate()
                .map(|(value, (level, &probability))| ScoreProbability {
                    value,
                    label: level.label,
                    probability,
                })
                .collect(),
            confidence,
        },
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::routes::decisions::questions::label_softmax;

    #[test]
    fn choice_values_are_written_like_python_json_dumps() {
        let question: DecisionQuestion = serde_json::from_value(json!({
            "type": "choice",
            "instructions": "Pick",
            "choices": [{"value": "a\"b\\c\n\u{1}é\u{2028}"}, {"value": true}],
        }))
        .unwrap();
        let names: Vec<_> = (read_question(&question).unwrap().options.into_iter())
            .map(|option| option.name)
            .collect();
        // `json.dumps('a"b\\c\n\x01é\u2028', ensure_ascii=False)` and
        // `json.dumps(True)`.
        assert_eq!(names, ["\"a\\\"b\\\\c\\n\\u0001é\u{2028}\"", "true"]);
    }

    #[test]
    fn softmax_matches_python_bit_for_bit() {
        // `label_softmax([-0.5, -1.25, -3.0])` in Python.
        assert_eq!(
            label_softmax(&[-0.5, -1.25, -3.0]),
            [0.6433137135229029, 0.3038798811878343, 0.05280640528926271]
        );
    }

    #[test]
    fn predicates_read_yes_then_no_and_scores_read_letters() {
        let predicate: DecisionQuestion =
            serde_json::from_value(json!({"type": "predicate", "instructions": "Q?"})).unwrap();
        let read = read_question(&predicate).unwrap();
        assert_eq!(read.question_type, QuestionType::Noul);
        assert_eq!(read.labels(), ["yes", "no"]);

        let score: DecisionQuestion = serde_json::from_value(json!({
            "type": "score",
            "instructions": "Rate",
            "levels": [{"label": "low"}, {"label": "high", "description": "very"}],
        }))
        .unwrap();
        let read = read_question(&score).unwrap();
        assert_eq!(read.labels(), ["A", "B"]);
        assert_eq!(
            read.prompt(),
            "Question: Rate\nA: low\nB: high - very\nAnswer with the letter of one option only."
        );
    }

    #[test]
    fn choices_beyond_the_labels_are_refused() {
        let choices: Vec<_> = (0..27).map(|i| json!({"value": i.to_string()})).collect();
        let question: DecisionQuestion = serde_json::from_value(
            json!({"type": "choice", "instructions": "Pick", "choices": choices}),
        )
        .unwrap();
        let error = read_question(&question).unwrap_err();
        assert!(
            format!("{error:?}").contains("This model supports at most 26 choices per question")
        );
    }
}
