// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;

use crate::backends::RequestFuncOutput;
use crate::benchmark::SpecDecodeStats;
use crate::config::BenchConfig;
use crate::error::Result;
use crate::metrics::{BenchmarkMetrics, MultiTurnMetrics};
use crate::multi_turn::ConversationOutput;

/// Build the result JSON object matching the Python output schema exactly.
///
/// Mirrors serve.py:1801-1947 result_json construction.
pub fn build_result_json(
    config: &BenchConfig,
    metrics: &BenchmarkMetrics,
    actual_output_lens: &[usize],
    outputs: &[RequestFuncOutput],
    benchmark_duration: f64,
    current_dt: &str,
    spec_decode_stats: Option<&SpecDecodeStats>,
) -> serde_json::Value {
    let mut result = serde_json::Map::new();

    // Setup
    result.insert("date".into(), Value::String(current_dt.to_string()));
    result.insert(
        "endpoint_type".into(),
        Value::String(config.backend.to_string()),
    );
    result.insert("backend".into(), Value::String(config.backend.to_string()));
    result.insert(
        "label".into(),
        config.label.as_ref().map(|l| Value::String(l.clone())).unwrap_or(Value::Null),
    );
    result.insert(
        "model_id".into(),
        Value::String(config.model.clone().unwrap_or_default()),
    );
    result.insert(
        "tokenizer_id".into(),
        config
            .tokenizer_id
            .as_ref()
            .map(|t| Value::String(t.clone()))
            .unwrap_or(Value::Null),
    );
    result.insert(
        "num_prompts".into(),
        Value::Number(config.num_prompts.into()),
    );
    result.insert(
        "max_model_len".into(),
        config.max_model_len.map(|v| Value::Number(v.into())).unwrap_or(Value::Null),
    );

    // Metadata
    if let Some(ref metadata) = config.metadata {
        for (k, v) in metadata {
            result.insert(k.clone(), Value::String(v.clone()));
        }
    }

    // Traffic
    if config.request_rate.is_infinite() {
        result.insert("request_rate".into(), Value::String("inf".to_string()));
    } else {
        result.insert(
            "request_rate".into(),
            serde_json::json!(config.request_rate),
        );
    }
    result.insert("burstiness".into(), serde_json::json!(config.burstiness));
    result.insert(
        "max_concurrency".into(),
        config.max_concurrency.map(|v| Value::Number(v.into())).unwrap_or(Value::Null),
    );

    let is_pooling = config.backend.is_pooling();

    // Benchmark results
    result.insert("duration".into(), serde_json::json!(benchmark_duration));
    result.insert("completed".into(), serde_json::json!(metrics.completed));
    result.insert("failed".into(), serde_json::json!(metrics.failed));
    result.insert(
        "total_input_tokens".into(),
        serde_json::json!(metrics.total_input),
    );
    if !is_pooling {
        result.insert(
            "total_output_tokens".into(),
            serde_json::json!(metrics.total_output),
        );
    }
    result.insert(
        "request_throughput".into(),
        serde_json::json!(metrics.request_throughput),
    );
    if config.goodput.is_empty() {
        result.insert("request_goodput".into(), Value::Null);
    } else {
        result.insert(
            "request_goodput".into(),
            serde_json::json!(metrics.request_goodput),
        );
    }
    if is_pooling {
        result.insert(
            "total_token_throughput".into(),
            serde_json::json!(metrics.total_token_throughput),
        );
    } else {
        result.insert(
            "input_throughput".into(),
            serde_json::json!(metrics.input_throughput),
        );
        result.insert(
            "output_throughput".into(),
            serde_json::json!(metrics.output_throughput),
        );
        result.insert(
            "total_token_throughput".into(),
            serde_json::json!(metrics.total_token_throughput),
        );
        result.insert(
            "max_output_tokens_per_s".into(),
            serde_json::json!(metrics.max_output_tokens_per_s),
        );
        result.insert(
            "max_concurrent_requests".into(),
            serde_json::json!(metrics.max_concurrent_requests),
        );
        // Inverse Real-Time Factor (for ASR benchmarks; 0.0 for generation)
        result.insert("rtfx".into(), serde_json::json!(0.0));
    }

    // Per-failure log (always emitted when there are failures, regardless of --save-detailed).
    // ttft/itl/output_tokens/start_time/latency are best-effort — zero/empty for failures
    // that occurred before any tokens were received. Non-zero values indicate a partial
    // failure (some tokens streamed before the error).
    if metrics.failed > 0 {
        result.insert(
            "failed_requests".into(),
            Value::Array(collect_failed_requests(outputs)),
        );
    }

    // Per-request data.
    // For pooling: Python always includes input_lens and errors (serve.py:1004-1013).
    // For generation: all per-request arrays are gated by --save-detailed.
    if is_pooling {
        result.insert(
            "input_lens".into(),
            serde_json::json!(outputs.iter().map(|o| o.prompt_len).collect::<Vec<_>>()),
        );
        result.insert(
            "errors".into(),
            serde_json::json!(outputs.iter().map(|o| &o.error).collect::<Vec<_>>()),
        );
    }
    if config.save_detailed {
        if !is_pooling {
            result.insert(
                "input_lens".into(),
                serde_json::json!(outputs.iter().map(|o| o.prompt_len).collect::<Vec<_>>()),
            );
            result.insert("output_lens".into(), serde_json::json!(actual_output_lens));
            result.insert(
                "ttfts".into(),
                serde_json::json!(outputs.iter().map(|o| o.ttft).collect::<Vec<_>>()),
            );
            result.insert(
                "itls".into(),
                serde_json::json!(outputs.iter().map(|o| &o.itl).collect::<Vec<_>>()),
            );
            result.insert(
                "generated_texts".into(),
                serde_json::json!(outputs.iter().map(|o| &o.generated_text).collect::<Vec<_>>()),
            );
            result.insert(
                "errors".into(),
                serde_json::json!(outputs.iter().map(|o| &o.error).collect::<Vec<_>>()),
            );
        }
        result.insert(
            "latencies".into(),
            serde_json::json!(outputs.iter().map(|o| o.latency).collect::<Vec<_>>()),
        );
        result.insert(
            "start_times".into(),
            serde_json::json!(outputs.iter().map(|o| o.start_time).collect::<Vec<_>>()),
        );
    }

    // Speculative decoding stats
    if let Some(stats) = spec_decode_stats {
        insert_spec_decode_stats(&mut result, stats);
    }

    // Per-metric stats (pooling only has e2el)
    if !is_pooling {
        add_metric_stats(
            &mut result,
            "ttft",
            metrics.mean_ttft_ms,
            metrics.median_ttft_ms,
            metrics.std_ttft_ms,
            &metrics.percentiles_ttft_ms,
            &config.selected_percentile_metrics,
        );
        add_metric_stats(
            &mut result,
            "tpot",
            metrics.mean_tpot_ms,
            metrics.median_tpot_ms,
            metrics.std_tpot_ms,
            &metrics.percentiles_tpot_ms,
            &config.selected_percentile_metrics,
        );
        add_metric_stats(
            &mut result,
            "itl",
            metrics.mean_itl_ms,
            metrics.median_itl_ms,
            metrics.std_itl_ms,
            &metrics.percentiles_itl_ms,
            &config.selected_percentile_metrics,
        );
    }
    add_metric_stats(
        &mut result,
        "e2el",
        metrics.mean_e2el_ms,
        metrics.median_e2el_ms,
        metrics.std_e2el_ms,
        &metrics.percentiles_e2el_ms,
        &config.selected_percentile_metrics,
    );

    // Steady-state metrics (null when not computed or scope gate failed).
    result.insert(
        "steady_state".into(),
        serde_json::to_value(&metrics.steady_state).unwrap_or(Value::Null),
    );

    Value::Object(result)
}

/// Build multi-turn result JSON.
pub fn build_multi_turn_result_json(
    config: &BenchConfig,
    mt_metrics: &MultiTurnMetrics,
    conversation_outputs: &[ConversationOutput],
    benchmark_duration: f64,
    current_dt: &str,
    spec_decode_stats: Option<&SpecDecodeStats>,
) -> serde_json::Value {
    let mut result = serde_json::Map::new();

    // Setup
    result.insert("date".into(), Value::String(current_dt.to_string()));
    result.insert("mode".into(), Value::String("multi_turn".to_string()));
    result.insert("backend".into(), Value::String(config.backend.to_string()));
    result.insert(
        "model_id".into(),
        Value::String(config.model.clone().unwrap_or_default()),
    );
    result.insert(
        "num_conversations".into(),
        Value::Number(config.num_prompts.into()),
    );
    result.insert(
        "max_model_len".into(),
        config.max_model_len.map(|v| Value::Number(v.into())).unwrap_or(Value::Null),
    );
    result.insert(
        "turns_per_conversation".into(),
        Value::Number(config.multi_turn_num_turns.into()),
    );
    result.insert(
        "multi_turn_concurrency".into(),
        serde_json::json!(
            config
                .multi_turn_concurrency
                .unwrap_or(config.max_concurrency.unwrap_or(config.num_prompts))
        ),
    );
    result.insert(
        "inter_turn_delay_ms".into(),
        Value::Number(config.multi_turn_delay_ms.into()),
    );
    if config.multi_turn_prefix_global_ratio > 0.0
        || config.multi_turn_prefix_conversation_ratio > 0.0
    {
        result.insert(
            "prefix_global_ratio".into(),
            serde_json::json!(config.multi_turn_prefix_global_ratio),
        );
        result.insert(
            "prefix_conversation_ratio".into(),
            serde_json::json!(config.multi_turn_prefix_conversation_ratio),
        );
        result.insert(
            "prefix_unique_ratio".into(),
            serde_json::json!(
                (1.0 - (config.multi_turn_prefix_global_ratio
                    + config.multi_turn_prefix_conversation_ratio))
                    .max(0.0)
            ),
        );
        result.insert("no_history_accumulation".into(), serde_json::json!(true));
    }

    // Metadata
    if let Some(ref metadata) = config.metadata {
        for (k, v) in metadata {
            result.insert(k.clone(), Value::String(v.clone()));
        }
    }

    // Conversation-level stats
    result.insert("duration".into(), serde_json::json!(benchmark_duration));
    result.insert(
        "conversations_completed".into(),
        serde_json::json!(mt_metrics.conversations_completed),
    );
    result.insert(
        "conversations_failed".into(),
        serde_json::json!(mt_metrics.conversations_failed),
    );
    result.insert(
        "avg_turns_completed".into(),
        serde_json::json!(mt_metrics.avg_turns_completed),
    );
    result.insert(
        "avg_conversation_duration_ms".into(),
        serde_json::json!(mt_metrics.avg_conversation_duration_ms),
    );

    // Overall metrics
    let overall = &mt_metrics.overall;
    result.insert("completed".into(), serde_json::json!(overall.completed));
    result.insert("failed".into(), serde_json::json!(overall.failed));

    // Per-failure log (always emitted when there are failures). Each entry is a single failed turn.
    if overall.failed > 0 {
        result.insert(
            "failed_requests".into(),
            Value::Array(collect_failed_turns(conversation_outputs)),
        );
    }
    result.insert(
        "total_input_tokens".into(),
        serde_json::json!(overall.total_input),
    );
    result.insert(
        "total_output_tokens".into(),
        serde_json::json!(overall.total_output),
    );
    result.insert(
        "request_throughput".into(),
        serde_json::json!(overall.request_throughput),
    );
    result.insert(
        "input_throughput".into(),
        serde_json::json!(overall.input_throughput),
    );
    result.insert(
        "output_throughput".into(),
        serde_json::json!(overall.output_throughput),
    );
    result.insert(
        "total_token_throughput".into(),
        serde_json::json!(overall.total_token_throughput),
    );

    // Overall per-metric stats
    add_metric_stats(
        &mut result,
        "ttft",
        overall.mean_ttft_ms,
        overall.median_ttft_ms,
        overall.std_ttft_ms,
        &overall.percentiles_ttft_ms,
        &config.selected_percentile_metrics,
    );
    add_metric_stats(
        &mut result,
        "tpot",
        overall.mean_tpot_ms,
        overall.median_tpot_ms,
        overall.std_tpot_ms,
        &overall.percentiles_tpot_ms,
        &config.selected_percentile_metrics,
    );
    add_metric_stats(
        &mut result,
        "itl",
        overall.mean_itl_ms,
        overall.median_itl_ms,
        overall.std_itl_ms,
        &overall.percentiles_itl_ms,
        &config.selected_percentile_metrics,
    );
    add_metric_stats(
        &mut result,
        "e2el",
        overall.mean_e2el_ms,
        overall.median_e2el_ms,
        overall.std_e2el_ms,
        &overall.percentiles_e2el_ms,
        &config.selected_percentile_metrics,
    );

    // Speculative decoding stats
    if let Some(stats) = spec_decode_stats {
        insert_spec_decode_stats(&mut result, stats);
    }

    // Per-turn metrics array
    let per_turn_json: Vec<Value> = mt_metrics
        .per_turn
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut turn = serde_json::Map::new();
            turn.insert("turn_index".into(), serde_json::json!(i));
            turn.insert(
                "num_samples".into(),
                serde_json::json!(m.completed + m.failed),
            );
            turn.insert("completed".into(), serde_json::json!(m.completed));
            turn.insert("failed".into(), serde_json::json!(m.failed));
            turn.insert(
                "total_input_tokens".into(),
                serde_json::json!(m.total_input),
            );
            turn.insert(
                "total_output_tokens".into(),
                serde_json::json!(m.total_output),
            );
            turn.insert(
                "request_throughput".into(),
                serde_json::json!(m.request_throughput),
            );
            turn.insert(
                "input_throughput".into(),
                serde_json::json!(m.input_throughput),
            );
            turn.insert(
                "output_throughput".into(),
                serde_json::json!(m.output_throughput),
            );
            turn.insert(
                "total_token_throughput".into(),
                serde_json::json!(m.total_token_throughput),
            );

            add_metric_stats(
                &mut turn,
                "ttft",
                m.mean_ttft_ms,
                m.median_ttft_ms,
                m.std_ttft_ms,
                &m.percentiles_ttft_ms,
                &config.selected_percentile_metrics,
            );
            add_metric_stats(
                &mut turn,
                "tpot",
                m.mean_tpot_ms,
                m.median_tpot_ms,
                m.std_tpot_ms,
                &m.percentiles_tpot_ms,
                &config.selected_percentile_metrics,
            );
            add_metric_stats(
                &mut turn,
                "itl",
                m.mean_itl_ms,
                m.median_itl_ms,
                m.std_itl_ms,
                &m.percentiles_itl_ms,
                &config.selected_percentile_metrics,
            );
            add_metric_stats(
                &mut turn,
                "e2el",
                m.mean_e2el_ms,
                m.median_e2el_ms,
                m.std_e2el_ms,
                &m.percentiles_e2el_ms,
                &config.selected_percentile_metrics,
            );

            Value::Object(turn)
        })
        .collect();

    result.insert("per_turn_metrics".into(), Value::Array(per_turn_json));

    Value::Object(result)
}

fn collect_failed_requests(outputs: &[RequestFuncOutput]) -> Vec<Value> {
    outputs
        .iter()
        .enumerate()
        .filter(|(_, o)| !o.success)
        .map(|(i, o)| {
            serde_json::json!({
                "index": i,
                "error": o.error,
                "prompt_len": o.prompt_len,
                "start_time": o.start_time,
                "latency": o.latency,
                "ttft": o.ttft,
                "output_tokens": o.output_tokens,
                "itl": o.itl,
            })
        })
        .collect()
}

fn collect_failed_turns(conversation_outputs: &[ConversationOutput]) -> Vec<Value> {
    let mut failed = Vec::new();
    for conv in conversation_outputs {
        for turn in &conv.turns {
            if !turn.request_output.success {
                let o = &turn.request_output;
                failed.push(serde_json::json!({
                    "conversation_id": conv.conversation_id,
                    "turn_index": turn.turn_index,
                    "error": o.error,
                    "prompt_len": o.prompt_len,
                    "start_time": o.start_time,
                    "latency": o.latency,
                    "ttft": o.ttft,
                    "output_tokens": o.output_tokens,
                    "itl": o.itl,
                }));
            }
        }
    }
    failed
}

fn insert_spec_decode_stats(result: &mut serde_json::Map<String, Value>, stats: &SpecDecodeStats) {
    result.insert(
        "spec_decode_acceptance_rate".into(),
        serde_json::json!(stats.acceptance_rate),
    );
    result.insert(
        "spec_decode_acceptance_length".into(),
        serde_json::json!(stats.acceptance_length),
    );
    result.insert(
        "spec_decode_num_drafts".into(),
        serde_json::json!(stats.num_drafts),
    );
    result.insert(
        "spec_decode_draft_tokens".into(),
        serde_json::json!(stats.draft_tokens),
    );
    result.insert(
        "spec_decode_accepted_tokens".into(),
        serde_json::json!(stats.accepted_tokens),
    );
    result.insert(
        "spec_decode_per_position_acceptance_rates".into(),
        serde_json::json!(stats.per_position_acceptance_rates),
    );
}

fn add_metric_stats(
    result: &mut serde_json::Map<String, Value>,
    name: &str,
    mean: f64,
    median: f64,
    std: f64,
    percentiles: &[(f64, f64)],
    selected: &[String],
) {
    if !selected.iter().any(|s| s == name) {
        return;
    }
    result.insert(format!("mean_{name}_ms"), serde_json::json!(mean));
    result.insert(format!("median_{name}_ms"), serde_json::json!(median));
    result.insert(format!("std_{name}_ms"), serde_json::json!(std));
    for (p, value) in percentiles {
        let p_str = if *p == p.floor() {
            format!("{}", *p as i64)
        } else {
            format!("{p}")
        };
        result.insert(format!("p{p_str}_{name}_ms"), serde_json::json!(value));
    }
}

/// Save result JSON to file (overwrite mode).
pub fn save_result(json: &Value, file_path: &str) -> Result<()> {
    let content = serde_json::to_string(json)?;
    std::fs::write(file_path, content)?;
    tracing::info!(path = file_path, "saved benchmark results");
    Ok(())
}

/// Append result JSON to file (JSONL format, matching Python's --append-result).
pub fn append_result(json: &Value, file_path: &str) -> Result<()> {
    use std::io::Write;
    let content = serde_json::to_string(json)?;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(file_path)?;
    // If file is non-empty, prepend a newline
    let meta = file.metadata()?;
    if meta.len() > 0 {
        file.write_all(b"\n")?;
    }
    file.write_all(content.as_bytes())?;
    tracing::info!(path = file_path, "appended benchmark results");
    Ok(())
}

/// Compute the result filename matching Python's logic.
pub fn compute_result_filename(config: &BenchConfig, model_id: &str, current_dt: &str) -> String {
    let base_model = model_id.split('/').next_back().unwrap_or(model_id);
    let max_conc_str =
        config.max_concurrency.map(|mc| format!("-concurrency{mc}")).unwrap_or_default();
    let label = config.label.as_deref().unwrap_or_else(|| config.backend.as_str());

    let file_name = if config.request_rate.is_infinite() {
        format!("{label}-infqps{max_conc_str}-{base_model}-{current_dt}.json")
    } else {
        format!(
            "{label}-{}qps{max_conc_str}-{base_model}-{current_dt}.json",
            config.request_rate
        )
    };

    if let Some(ref explicit) = config.result_filename {
        if let Some(ref dir) = config.result_dir {
            return format!("{dir}/{explicit}");
        }
        return explicit.clone();
    }

    if let Some(ref dir) = config.result_dir {
        format!("{dir}/{file_name}")
    } else {
        file_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_spec_decode_stats() {
        let stats = SpecDecodeStats {
            num_drafts: 1000,
            draft_tokens: 3000,
            accepted_tokens: 2400,
            acceptance_rate: 80.0,
            acceptance_length: 3.4,
            per_position_acceptance_rates: vec![0.9, 0.8, 0.7],
        };

        let mut result = serde_json::Map::new();
        insert_spec_decode_stats(&mut result, &stats);

        assert_eq!(result["spec_decode_acceptance_rate"], 80.0);
        assert_eq!(result["spec_decode_acceptance_length"], 3.4);
        assert_eq!(result["spec_decode_num_drafts"], 1000);
        assert_eq!(result["spec_decode_draft_tokens"], 3000);
        assert_eq!(result["spec_decode_accepted_tokens"], 2400);
        assert_eq!(
            result["spec_decode_per_position_acceptance_rates"],
            serde_json::json!([0.9, 0.8, 0.7])
        );
    }

    #[test]
    fn test_insert_spec_decode_stats_empty_positions() {
        let stats = SpecDecodeStats {
            num_drafts: 500,
            draft_tokens: 1500,
            accepted_tokens: 1200,
            acceptance_rate: 80.0,
            acceptance_length: 3.4,
            per_position_acceptance_rates: vec![],
        };

        let mut result = serde_json::Map::new();
        insert_spec_decode_stats(&mut result, &stats);

        assert_eq!(
            result["spec_decode_per_position_acceptance_rates"],
            serde_json::json!([])
        );
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_collect_failed_requests_empty_when_all_success() {
        let outputs = vec![
            RequestFuncOutput {
                success: true,
                ..Default::default()
            },
            RequestFuncOutput {
                success: true,
                ..Default::default()
            },
        ];
        assert!(collect_failed_requests(&outputs).is_empty());
    }

    #[test]
    fn test_collect_failed_requests_preserves_index_and_fields() {
        let outputs = vec![
            RequestFuncOutput {
                success: true,
                ..Default::default()
            },
            // Pre-stream failure: no tokens received.
            RequestFuncOutput {
                success: false,
                error: "connection reset".into(),
                prompt_len: 128,
                start_time: 1.5,
                latency: 0.0,
                ttft: 0.0,
                output_tokens: 0,
                itl: vec![],
                ..Default::default()
            },
            RequestFuncOutput {
                success: true,
                ..Default::default()
            },
            // Partial failure: streamed 3 tokens before timing out.
            RequestFuncOutput {
                success: false,
                error: "timeout".into(),
                prompt_len: 64,
                start_time: 2.25,
                latency: 30.0,
                ttft: 0.5,
                output_tokens: 3,
                itl: vec![0.1, 0.12],
                ..Default::default()
            },
        ];

        let failed = collect_failed_requests(&outputs);
        assert_eq!(failed.len(), 2);

        assert_eq!(failed[0]["index"], 1);
        assert_eq!(failed[0]["error"], "connection reset");
        assert_eq!(failed[0]["prompt_len"], 128);
        assert_eq!(failed[0]["start_time"], 1.5);
        assert_eq!(failed[0]["latency"], 0.0);
        assert_eq!(failed[0]["ttft"], 0.0);
        assert_eq!(failed[0]["output_tokens"], 0);
        assert_eq!(failed[0]["itl"], serde_json::json!([]));

        assert_eq!(failed[1]["index"], 3);
        assert_eq!(failed[1]["error"], "timeout");
        assert_eq!(failed[1]["latency"], 30.0);
        assert_eq!(failed[1]["ttft"], 0.5);
        assert_eq!(failed[1]["output_tokens"], 3);
        assert_eq!(failed[1]["itl"], serde_json::json!([0.1, 0.12]));
    }
}
