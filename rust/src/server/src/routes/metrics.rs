// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use axum::Extension;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use thiserror_ext::AsReport;
use vllm_metrics::METRICS;

use crate::middleware::ListenerRank;

const OPENMETRICS_CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// Label that carries the engine index (data-parallel rank) on per-engine
/// series.
const ENGINE_LABEL: &str = "engine";

/// Render the metrics registry. On a data-parallel rank listener, per-engine
/// series of other engines are left out, so a router that scrapes each rank's
/// port sees that rank's numbers alone, as from a frontend that serves only
/// that rank. Series without an engine label are kept.
pub(crate) async fn scrape(listener_rank: Option<Extension<ListenerRank>>) -> Response {
    match METRICS.render() {
        Ok(body) => {
            let body = match listener_rank {
                Some(Extension(ListenerRank(rank))) => retain_engine(&body, rank),
                None => body,
            };
            (
                [(
                    CONTENT_TYPE,
                    HeaderValue::from_static(OPENMETRICS_CONTENT_TYPE),
                )],
                body,
            )
                .into_response()
        }

        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to render metrics: {}", error.as_report()),
        )
            .into_response(),
    }
}

/// Drop the sample lines whose `engine` label names an engine other than
/// `rank`. Comment lines (`# HELP`, `# TYPE`, `# EOF`) and samples without an
/// `engine` label are kept.
fn retain_engine(text: &str, rank: u32) -> String {
    let rank = rank.to_string();
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if line.starts_with('#') || engine_label(line).is_none_or(|engine| engine == rank) {
            out.push_str(line);
        }
    }
    out
}

/// Return the raw value of the `engine` label of a sample line, if any.
fn engine_label(line: &str) -> Option<&str> {
    let mut rest = &line[line.find('{')? + 1..];
    loop {
        rest = rest.trim_start_matches([',', ' ']);
        if rest.is_empty() || rest.starts_with('}') {
            return None;
        }
        let eq = rest.find('=')?;
        let name = rest[..eq].trim();
        let value = rest[eq + 1..].strip_prefix('"')?;
        let end = closing_quote(value)?;
        if name == ENGINE_LABEL {
            return Some(&value[..end]);
        }
        rest = &value[end + 1..];
    }
}

/// Byte index of the quote that closes a label value, skipping escapes.
fn closing_quote(value: &str) -> Option<usize> {
    let mut escaped = false;
    for (index, ch) in value.char_indices() {
        match ch {
            '\\' if !escaped => escaped = true,
            '"' if !escaped => return Some(index),
            _ => escaped = false,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const RENDERED: &str = "\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name=\"m\",engine=\"0\"} 0.5
vllm:kv_cache_usage_perc{model_name=\"m\",engine=\"1\"} 0.25
# HELP vllm:request_success Finished requests.
# TYPE vllm:request_success counter
vllm:request_success_total{model_name=\"m, \\\"engine=\\\"0\",finished_reason=\"stop\",engine=\"1\"} 3
vllm:request_success_total{finished_reason=\"stop\",engine=\"10\",model_name=\"m\"} 4
# HELP vllm:http_requests HTTP requests.
# TYPE vllm:http_requests counter
vllm:http_requests_total{method=\"GET\"} 7
process_open_fds 12
# EOF
";

    #[test]
    fn keeps_the_rank_and_engine_independent_series() {
        let filtered = retain_engine(RENDERED, 1);
        assert!(!filtered.contains("engine=\"0\"} 0.5"));
        assert!(filtered.contains("vllm:kv_cache_usage_perc{model_name=\"m\",engine=\"1\"} 0.25"));
        assert!(filtered.contains("finished_reason=\"stop\",engine=\"1\"} 3"));
        assert!(
            !filtered.contains("engine=\"10\""),
            "rank 1 must not match engine 10"
        );
        assert!(filtered.contains("vllm:http_requests_total{method=\"GET\"} 7"));
        assert!(filtered.contains("process_open_fds 12"));
        assert!(filtered.contains("# TYPE vllm:kv_cache_usage_perc gauge"));
        assert!(filtered.ends_with("# EOF\n"));
    }

    #[test]
    fn drops_only_the_other_engines_samples() {
        // Rank 0 keeps its one per-engine sample and drops the three of
        // engines 1 and 10.
        assert_eq!(
            retain_engine(RENDERED, 0).lines().count(),
            RENDERED.lines().count() - 3
        );
    }

    #[test]
    fn reads_the_engine_label_past_escaped_quotes() {
        let line = "x{model_name=\"a \\\"engine=\\\"9\\\" b\",engine=\"2\"} 1";
        assert_eq!(engine_label(line), Some("2"));
        assert_eq!(engine_label("x{a=\"1\"} 1"), None);
        assert_eq!(engine_label("x 1"), None);
    }
}
