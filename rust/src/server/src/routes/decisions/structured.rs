// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structured decisions API (`POST /v1/systemone`): questions keyed by id
//! about a state, answered with next-token reads.
//!
//! Original Python definitions: `vllm/entrypoints/generate/structured_decisions/`
//! and the question types' `parse_options` and `answer` in
//! `vllm/entrypoints/openai/decisions/question_types.py`.

use std::collections::{BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_json_fmt::JsonFormat;

use super::next_token::{MAX_QUESTIONS, QuestionRead};
use super::questions::{Question, QuestionOption, QuestionType, argmax};
use crate::error::{ApiError, bail_invalid_request, invalid_request};

/// `POST /v1/systemone` request body.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredDecisionRequest {
    pub model: Option<String>,
    /// What the questions are about: text or JSON.
    pub state: Value,
    /// Question id to question, asked in this order.
    pub questions: IndexMap<String, QuestionSpec>,
    /// Context placed ahead of the questions, as a system message.
    pub instructions: Option<String>,
    pub images: Option<Vec<String>>,
    pub audio: Option<Vec<String>>,
    pub videos: Option<Vec<String>>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    /// Answer the questions jointly; diffusion models only.
    #[serde(default)]
    pub joint: bool,
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub request_id: Option<String>,
}

/// One question of a request. Python's `QuestionSpec`.
#[derive(Debug, Clone, Deserialize)]
pub struct QuestionSpec {
    #[serde(rename = "type")]
    pub question_type: String,
    #[serde(default)]
    pub instructions: Value,
    /// Shaped by the type; [`parse_options`] reads it.
    #[serde(default)]
    pub criteria: Value,
    /// Fields the spec does not declare, which Python refuses by name.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl StructuredDecisionRequest {
    /// Refuse the request's media, like Python's `_media`: a read renders
    /// images only, and only for a multimodal model, which the Rust frontend
    /// does not do yet.
    pub fn check_media(&self, multimodal: bool) -> Result<(), ApiError> {
        let media: BTreeSet<_> = [
            ("image", &self.images),
            ("audio", &self.audio),
            ("video", &self.videos),
        ]
        .into_iter()
        .filter(|(_, urls)| urls.as_ref().is_some_and(|urls| !urls.is_empty()))
        .map(|(modality, _)| modality)
        .collect();
        let unsupported: Vec<_> = (media.iter())
            .filter(|&&modality| !(multimodal && modality == "image"))
            .copied()
            .collect();
        if !unsupported.is_empty() {
            bail_invalid_request!("this model does not take {} inputs", unsupported.join(", "));
        }
        if !media.is_empty() {
            bail_invalid_request!(
                "the Rust frontend does not read images on /v1/systemone yet; set \
                 VLLM_USE_RUST_FRONTEND=0 to use the Python frontend"
            );
        }
        Ok(())
    }

    /// The state as the model reads it: text as is, anything else as Python's
    /// `json.dumps(state, ensure_ascii=False)`.
    pub fn state_text(&self) -> String {
        match &self.state {
            Value::String(text) => text.clone(),
            state => JsonFormat::new()
                .comma(", ")
                .expect("a valid separator")
                .colon(": ")
                .expect("a valid separator")
                .ascii(false)
                .format_to_string(state)
                .expect("a JSON value serializes"),
        }
    }
}

/// `POST /v1/systemone` response body.
#[derive(Debug, Clone, Serialize)]
pub struct StructuredDecisionResponse {
    pub id: String,
    /// Always `structured_decision`.
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    /// Question id to answer, in request order.
    pub answers: IndexMap<String, StructuredAnswer>,
    pub usage: DecisionUsage,
    pub diagnostics: IndexMap<String, QuestionDiagnostics>,
    /// Always `null`: only diffusion models read jointly.
    pub joint: (),
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// One question's answer, as the question type's Python `answer` writes it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StructuredAnswer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        /// The chosen label's probability over the whole vocabulary.
        confidence: f64,
    },
    Noul {
        /// The probability of yes.
        noul: f64,
        probabilities: NoulProbabilities,
        confidence: f64,
    },
    Score {
        /// The expected level, 0-indexed.
        score: f64,
        /// Level to its name.
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NoulProbabilities {
    pub yes: f64,
    pub no: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QuestionDiagnostics {
    /// The labels' total probability over the whole vocabulary.
    pub label_mass: f64,
    /// Whether the greedy reply's first token is one of the labels.
    pub argmax_is_label: bool,
}

impl From<&QuestionRead> for QuestionDiagnostics {
    fn from(read: &QuestionRead) -> Self {
        Self {
            label_mass: read.label_mass,
            argmax_is_label: read.argmax_is_label,
        }
    }
}

/// Parse the request's questions, in order, as Python's `parse_questions`.
pub(super) fn parse_questions(
    questions: &IndexMap<String, QuestionSpec>,
) -> Result<Vec<(String, Question)>, ApiError> {
    if questions.is_empty() {
        bail_invalid_request!("questions: needs at least one question");
    }
    if questions.len() > MAX_QUESTIONS {
        bail_invalid_request!("questions: at most {MAX_QUESTIONS} for this model");
    }
    (questions.iter())
        .map(|(id, spec)| {
            let question = build_question(id, spec).map_err(|e| invalid_request!("{e}"))?;
            Ok((id.clone(), question))
        })
        .collect()
}

/// One question from its spec. Python's `build_question`, after its
/// `parse_questions` refuses undeclared fields.
fn build_question(id: &str, spec: &QuestionSpec) -> Result<Question, String> {
    let qid = format!("'{id}'");
    if !spec.extra.is_empty() {
        let mut extra: Vec<_> = spec.extra.keys().map(|key| format!("'{key}'")).collect();
        extra.sort();
        return Err(format!(
            "question {qid}: unknown field(s) [{}]",
            extra.join(", ")
        ));
    }
    if id.is_empty() {
        return Err("question ids must be non-empty".to_owned());
    }
    let Some(question_type) = QuestionType::from_name(&spec.question_type) else {
        let supported: Vec<_> =
            QuestionType::ALL.iter().map(|t| format!("'{}'", t.name())).collect();
        return Err(format!(
            "unknown question type '{}'; supported: [{}]",
            spec.question_type,
            supported.join(", ")
        ));
    };
    let options = parse_options(question_type, &qid, &spec.criteria)?;
    if options.is_empty() {
        return Err(format!("question {qid}: needs at least one option"));
    }
    let mut names = HashSet::new();
    if !options.iter().all(|option| names.insert(&option.name)) {
        return Err(format!("question {qid}: duplicate option names"));
    }
    let limit = question_type.label_set().len();
    if options.len() > limit {
        return Err(format!(
            "question {qid}: at most {limit} options for this model"
        ));
    }
    let instructions = match &spec.instructions {
        Value::Null => String::new(),
        instructions => python_str(instructions)
            .ok_or_else(|| format!("question {qid}: instructions must be text or a number"))?,
    };
    Ok(Question {
        question_type,
        instructions,
        options,
    })
}

/// The options a type reads from its `criteria`. Python's `parse_options`.
fn parse_options(
    question_type: QuestionType,
    qid: &str,
    criteria: &Value,
) -> Result<Vec<QuestionOption>, String> {
    let description = |value: &Value| match value {
        Value::Null => Ok(None),
        value => python_str(value)
            .map(Some)
            .ok_or_else(|| format!("question {qid}: descriptions must be text or numbers")),
    };
    match question_type {
        // An option name maps to a description, or to null.
        QuestionType::Choice => {
            let Some(criteria) = criteria.as_object().filter(|criteria| !criteria.is_empty())
            else {
                return Err(format!(
                    "question {qid}: choice criteria must map option names to a description or null"
                ));
            };
            (criteria.iter())
                .map(|(name, value)| {
                    Ok(QuestionOption {
                        name: name.clone(),
                        description: description(value)?,
                    })
                })
                .collect()
        }
        // Optional descriptions of what true and false mean.
        QuestionType::Noul => {
            let empty = Map::new();
            let criteria = match criteria {
                Value::Null => &empty,
                Value::Object(criteria)
                    if criteria.keys().all(|key| key == "true" || key == "false") =>
                {
                    criteria
                }
                _ => {
                    return Err(format!(
                        "question {qid}: noul criteria must be an object with true and false"
                    ));
                }
            };
            let side = |name: &str, key: &str| -> Result<QuestionOption, String> {
                Ok(QuestionOption {
                    name: name.to_owned(),
                    description: description(criteria.get(key).unwrap_or(&Value::Null))?,
                })
            };
            Ok(vec![side("yes", "true")?, side("no", "false")?])
        }
        // The ordered level names, from the first to the last level.
        QuestionType::Score => {
            let Some(levels) = criteria.as_array().filter(|levels| levels.len() >= 2) else {
                return Err(format!(
                    "question {qid}: score criteria must be an ordered list of levels"
                ));
            };
            (levels.iter())
                .map(|level| {
                    let name = python_str(level)
                        .ok_or_else(|| format!("question {qid}: levels must be text or numbers"))?;
                    Ok(QuestionOption {
                        name,
                        description: None,
                    })
                })
                .collect()
        }
    }
}

/// A question's answer, as the question type's Python `answer` writes it.
pub(super) fn answer(question: &Question, read: &QuestionRead) -> StructuredAnswer {
    let probs = &read.probs;
    let top = argmax(probs);
    let confidence = probs[top] * read.label_mass;
    let names = question.options.iter().map(|option| option.name.clone());
    let levels = || (0..probs.len()).map(|level| level.to_string());
    match question.question_type {
        QuestionType::Choice => StructuredAnswer::Choice {
            choice: question.options[top].name.clone(),
            probabilities: names.zip(probs.iter().copied()).collect(),
            confidence,
        },
        QuestionType::Noul => StructuredAnswer::Noul {
            noul: probs[0],
            probabilities: NoulProbabilities {
                yes: probs[0],
                no: probs[1],
            },
            confidence,
        },
        QuestionType::Score => StructuredAnswer::Score {
            score: probs.iter().enumerate().map(|(level, p)| level as f64 * p).sum(),
            legend: levels().zip(names).collect(),
            probabilities: levels().zip(probs.iter().copied()).collect(),
            confidence,
        },
    }
}

/// Python's `str()` of a JSON scalar: text as is, `True`, `False`, `None`.
/// Numbers print as in the JSON state, which matches Python except for floats
/// in exponent form. `None` for arrays and objects.
fn python_str(value: &Value) -> Option<String> {
    Some(match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => number.to_string(),
        Value::Array(_) | Value::Object(_) => return None,
    })
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    use serde_json::json;

    fn spec(spec: Value) -> QuestionSpec {
        serde_json::from_value(spec).expect("a question spec")
    }

    /// The prompt of one question spec, or its error, as Python's
    /// `build_question` and `prompt` give them.
    fn read(id: &str, question: Value) -> String {
        match build_question(id, &spec(question)) {
            Ok(question) => question.prompt(),
            Err(error) => format!("ERR {error}"),
        }
    }

    #[test]
    fn questions_match_python() {
        let reads = [
            read(
                "team",
                json!({"type": "choice", "instructions": "Which team?",
                       "criteria": {"billing": null, "shipping": "parcels", "security": 7}}),
            ),
            read(
                "urgent",
                json!({"type": "noul", "instructions": "Is money moving?"}),
            ),
            read(
                "urgent2",
                json!({"type": "noul", "instructions": "",
                       "criteria": {"true": "it moves", "false": null}}),
            ),
            read(
                "tone",
                json!({"type": "score", "instructions": "How upset?",
                       "criteria": ["calm", 2, 2.5, true, null]}),
            ),
            read(
                "n",
                json!({"type": "choice", "instructions": 5, "criteria": {"a": null}}),
            ),
            read(
                "",
                json!({"type": "choice", "instructions": "x", "criteria": {"a": null}}),
            ),
            read("q", json!({"type": "nope", "instructions": "x"})),
            read(
                "q",
                json!({"type": "choice", "instructions": "x", "criteria": {}}),
            ),
            read(
                "q",
                json!({"type": "choice", "instructions": "x", "criteria": ["a"]}),
            ),
            read(
                "q",
                json!({"type": "noul", "instructions": "x", "criteria": ["yes", "no"]}),
            ),
            read(
                "q",
                json!({"type": "noul", "instructions": "x", "criteria": {"maybe": "x"}}),
            ),
            read(
                "q",
                json!({"type": "score", "instructions": "x", "criteria": ["calm"]}),
            ),
            read(
                "q",
                json!({"type": "score", "instructions": "x", "criteria": ["a", "a"]}),
            ),
            read(
                "q",
                json!({"type": "score", "instructions": "x",
                       "criteria": (0..11).map(|i| i.to_string()).collect::<Vec<_>>()}),
            ),
            read(
                "it's",
                json!({"type": "score", "instructions": "x", "criteria": ["a"]}),
            ),
            read(
                "q",
                json!({"type": "choice", "criteria": {"a": null}, "depends_on": ["x"]}),
            ),
        ];
        // The Python output of the same specs, but for the last one, whose
        // unknown field pydantic reports before `build_question`, and for
        // `it's`, which Python quotes with double quotes.
        expect![[r#"
            [
                "Question: Which team?\nA: billing\nB: shipping - parcels\nC: security - 7\nAnswer with the letter of one option only.",
                "Question: Is money moving?\nyes\nno\nAnswer with yes or no only.",
                "yes: it moves\nno\nAnswer with yes or no only.",
                "Question: How upset?\n0: calm\n1: 2\n2: 2.5\n3: True\n4: None\nAnswer with the number of one level only.",
                "Question: 5\nA: a\nAnswer with the letter of one option only.",
                "ERR question ids must be non-empty",
                "ERR unknown question type 'nope'; supported: ['choice', 'noul', 'score']",
                "ERR question 'q': choice criteria must map option names to a description or null",
                "ERR question 'q': choice criteria must map option names to a description or null",
                "ERR question 'q': noul criteria must be an object with true and false",
                "ERR question 'q': noul criteria must be an object with true and false",
                "ERR question 'q': score criteria must be an ordered list of levels",
                "ERR question 'q': duplicate option names",
                "ERR question 'q': at most 10 options for this model",
                "ERR question 'it's': score criteria must be an ordered list of levels",
                "ERR question 'q': unknown field(s) ['depends_on']",
            ]
        "#]]
        .assert_debug_eq(&reads);
    }

    #[test]
    fn state_and_scalars_print_like_python() {
        let request: StructuredDecisionRequest = serde_json::from_value(json!({
            "state": {"team": "billing", "n": [1, 2.5, null, true], "é": "ü"},
            "questions": {},
        }))
        .unwrap();
        // `json.dumps(state, ensure_ascii=False)`.
        assert_eq!(
            request.state_text(),
            r#"{"team": "billing", "n": [1, 2.5, null, true], "é": "ü"}"#
        );
        // Python's `str()` of the same levels, but Python writes `1e-05` for `1e-5`.
        let levels: Vec<_> =
            (json!([2, 2.5, 0.0001, 1e-5, 1e16, true, null]).as_array().unwrap().iter())
                .map(|level| python_str(level).unwrap())
                .collect();
        assert_eq!(
            levels,
            ["2", "2.5", "0.0001", "0.00001", "1e+16", "True", "None"]
        );
    }

    #[test]
    fn answers_are_shaped_like_python() {
        let read = |probs: Vec<f64>| QuestionRead {
            probs,
            label_mass: 0.5,
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            cache_write_tokens: 0,
            argmax_is_label: true,
        };
        let choice = build_question(
            "team",
            &spec(json!({"type": "choice", "criteria": {"billing": null, "shipping": null}})),
        )
        .unwrap();
        let noul = build_question("urgent", &spec(json!({"type": "noul"}))).unwrap();
        let score = build_question(
            "tone",
            &spec(json!({"type": "score", "criteria": ["calm", "angry"]})),
        )
        .unwrap();
        let answers = [
            answer(&choice, &read(vec![0.25, 0.75])),
            answer(&noul, &read(vec![0.75, 0.25])),
            answer(&score, &read(vec![0.25, 0.75])),
        ];
        expect![[r#"
            [{"type":"choice","choice":"shipping","probabilities":{"billing":0.25,"shipping":0.75},"confidence":0.375},{"type":"noul","noul":0.75,"probabilities":{"yes":0.75,"no":0.25},"confidence":0.375},{"type":"score","score":0.75,"legend":{"0":"calm","1":"angry"},"probabilities":{"0":0.25,"1":0.75},"confidence":0.375}]"#]]
        .assert_eq(&serde_json::to_string(&answers).unwrap());
    }
}
