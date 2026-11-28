// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Question types: how a question and its labeled options are written for the
//! model, and how label probabilities are read back.
//!
//! Original Python definition: `vllm/entrypoints/openai/decisions/question_types.py`.
//! A read's prompt must match Python's byte for byte, so the texts here are
//! Python's.

/// A choice labels its options `A` to `Z`, in order.
const LETTERS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];
/// A score labels its levels `0` to `9`, so a level's label is its value.
const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// How a question is put to the model and which labels its reply may start
/// with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QuestionType {
    /// Pick one option, labeled with a letter.
    Choice,
    /// Yes or no.
    Noul,
    /// Rate on an ordered scale, labeled with a digit.
    Score,
}

impl QuestionType {
    /// Every type, in Python's registration order.
    pub(super) const ALL: [Self; 3] = [Self::Choice, Self::Noul, Self::Score];

    /// The name requests use for the type.
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "noul",
            Self::Score => "score",
        }
    }

    /// The type a request names, if any.
    pub(super) fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|question_type| question_type.name() == name)
    }

    /// The labels of a question's options, in option order.
    pub(super) fn label_set(self) -> &'static [&'static str] {
        match self {
            Self::Choice => &LETTERS,
            Self::Noul => &["yes", "no"],
            Self::Score => &DIGITS,
        }
    }

    fn reply_instruction(self) -> &'static str {
        match self {
            Self::Choice => "Answer with the letter of one option only.",
            Self::Noul => "Answer with yes or no only.",
            Self::Score => "Answer with the number of one level only.",
        }
    }

    fn reply_line(self, label: &str, option: &QuestionOption) -> String {
        let description = option.description.as_deref().filter(|d| !d.is_empty());
        match (self, description) {
            // A side without a description still shows its label. Without the
            // label in the prompt, models answer "Yes" or "No" instead.
            (Self::Noul, Some(description)) => format!("{label}: {description}"),
            (Self::Noul, None) => label.to_owned(),
            (_, Some(description)) => format!("{label}: {} - {description}", option.name),
            (_, None) => format!("{label}: {}", option.name),
        }
    }
}

/// One option of a question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QuestionOption {
    pub name: String,
    pub description: Option<String>,
}

/// One question, read as one engine request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Question {
    pub question_type: QuestionType,
    pub instructions: String,
    pub options: Vec<QuestionOption>,
}

impl Question {
    /// The labels of the options, in option order.
    pub(super) fn labels(&self) -> &'static [&'static str] {
        &self.question_type.label_set()[..self.options.len()]
    }

    /// The question as the model reads it, after the state.
    pub(super) fn prompt(&self) -> String {
        let mut lines: Vec<String> = (self.labels().iter())
            .zip(&self.options)
            .map(|(label, option)| self.question_type.reply_line(label, option))
            .collect();
        lines.push(self.question_type.reply_instruction().to_owned());
        let body = lines.join("\n");
        if self.instructions.is_empty() {
            body
        } else {
            format!("Question: {}\n{body}", self.instructions)
        }
    }
}

/// Softmax of the label logprobs.
pub(super) fn label_softmax(logprobs: &[f64]) -> Vec<f64> {
    let top = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logprobs.iter().map(|logprob| (logprob - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// The index of the largest value, the first one on a tie, like Python's
/// `max(range(n), key=...)`.
pub(super) fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, &value) in values.iter().enumerate().skip(1) {
        if value > values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::*;

    fn option(name: &str, description: Option<&str>) -> QuestionOption {
        QuestionOption {
            name: name.to_owned(),
            description: description.map(str::to_owned),
        }
    }

    #[test]
    fn prompts_match_python() {
        let choice = Question {
            question_type: QuestionType::Choice,
            instructions: "Which department should handle this?".to_owned(),
            options: vec![
                option("\"returns\"", Some("")),
                option("\"billing\"", Some("charges")),
            ],
        };
        expect![[r#"
            Question: Which department should handle this?
            A: "returns"
            B: "billing" - charges
            Answer with the letter of one option only."#]]
        .assert_eq(&choice.prompt());

        let noul = Question {
            question_type: QuestionType::Noul,
            instructions: "Is the screen broken?".to_owned(),
            options: vec![option("yes", None), option("no", Some("it works"))],
        };
        expect![[r#"
            Question: Is the screen broken?
            yes
            no: it works
            Answer with yes or no only."#]]
        .assert_eq(&noul.prompt());

        let score = Question {
            question_type: QuestionType::Score,
            instructions: String::new(),
            options: vec![option("low", None), option("high", None)],
        };
        expect![[r#"
            0: low
            1: high
            Answer with the number of one level only."#]]
        .assert_eq(&score.prompt());
    }

    #[test]
    fn softmax_and_argmax_follow_python() {
        let probs = label_softmax(&[0.5_f64.ln(), 0.25_f64.ln(), 0.25_f64.ln()]);
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 0.5).abs() < 1e-12 && (probs[1] - 0.25).abs() < 1e-12);
        assert_eq!(argmax(&[0.25, 0.5, 0.5]), 1);
        assert_eq!(argmax(&[0.4]), 0);
    }
}
