// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde::Serialize;

/// Output tokens `[start, end)` were sampled by weights labelled `version`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WeightVersionSpan {
    pub version: Arc<str>,
    pub start: usize,
    pub end: usize,
}

/// Weight-version labels of output tokens, as runs of consecutive tokens that
/// share a label (`None` where the engine reported no label).
///
/// Each output carries the labels of its own tokens, so a consumer appends the
/// labels of every output it keeps and builds the spans once, when the request
/// finishes, like [`AuxOutputRows`](crate::AuxOutputRows). Text decoding drops the labels of tokens
/// cut off by a stop string the same way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightVersions {
    runs: Vec<(Option<Arc<str>>, usize)>,
}

impl WeightVersions {
    /// Labels of `num_tokens` tokens sampled under `version`.
    pub fn new(version: Option<Arc<str>>, num_tokens: usize) -> Self {
        Self {
            runs: (num_tokens > 0).then_some((version, num_tokens)).into_iter().collect(),
        }
    }

    /// Append the labels of a later output.
    pub fn append(&mut self, later: Self) {
        for (version, num_tokens) in later.runs {
            match self.runs.last_mut() {
                Some((last, count)) if *last == version => *count += num_tokens,
                _ => self.runs.push((version, num_tokens)),
            }
        }
    }

    /// Drop the labels of up to `num_tokens` tokens from the end, for the
    /// tokens a stop string cut off.
    pub fn drop_trailing(&mut self, mut num_tokens: usize) {
        while num_tokens > 0
            && let Some((_, count)) = self.runs.last_mut()
        {
            let dropped = num_tokens.min(*count);
            *count -= dropped;
            num_tokens -= dropped;
            if *count == 0 {
                self.runs.pop();
            }
        }
    }

    /// Spans over the tokens, like Python's `_weight_version_spans()`: a span
    /// starts where the label changes and an unlabelled token belongs to the
    /// span before it. `[]` for no tokens, `None` if no label was reported.
    pub fn spans(self) -> Option<Vec<WeightVersionSpan>> {
        let mut spans: Vec<WeightVersionSpan> = Vec::new();
        let mut num_tokens = 0;
        for (version, count) in self.runs {
            if let Some(version) = version
                && spans.last().is_none_or(|last| last.version != version)
            {
                if let Some(last) = spans.last_mut() {
                    last.end = num_tokens;
                }
                spans.push(WeightVersionSpan {
                    version,
                    start: num_tokens,
                    end: num_tokens,
                });
            }
            num_tokens += count;
        }
        if spans.is_empty() && num_tokens > 0 {
            return None;
        }
        if let Some(last) = spans.last_mut() {
            last.end = num_tokens;
        }
        Some(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(runs: &[(Option<&str>, usize)]) -> WeightVersions {
        let mut labels = WeightVersions::default();
        for &(version, num_tokens) in runs {
            labels.append(WeightVersions::new(version.map(Arc::from), num_tokens));
        }
        labels
    }

    fn span(version: &str, start: usize, end: usize) -> WeightVersionSpan {
        WeightVersionSpan {
            version: version.into(),
            start,
            end,
        }
    }

    #[test]
    fn spans_start_where_the_label_changes() {
        assert_eq!(labels(&[]).spans(), Some(vec![]));
        assert_eq!(labels(&[(None, 3)]).spans(), None);
        // Unlabelled tokens belong to the span before them; leading ones to none.
        assert_eq!(
            labels(&[
                (None, 1),
                (Some("v1"), 2),
                (None, 1),
                (Some("v1"), 1),
                (Some("v2"), 2),
            ])
            .spans(),
            Some(vec![span("v1", 1, 5), span("v2", 5, 7)])
        );
    }

    #[test]
    fn drop_trailing_cuts_labels_across_outputs() {
        let mut all = labels(&[(Some("v1"), 2), (Some("v2"), 3)]);
        all.drop_trailing(2);
        assert_eq!(
            all.clone().spans(),
            Some(vec![span("v1", 0, 2), span("v2", 2, 3)])
        );
        all.drop_trailing(2);
        assert_eq!(all.clone().spans(), Some(vec![span("v1", 0, 1)]));
        all.drop_trailing(5);
        assert_eq!(all.spans(), Some(vec![]));
    }
}
