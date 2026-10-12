// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::unified::KimiK3UnifiedParser;
use vllm_tokenizer::Tokenizer;
use vllm_tokenizer::test_utils::TestTokenizer;

mod utils;
use utils::{attributed_chunks, feed_unified_parser};

const CHUNK_CHARS: usize = 7;
const LONG_REASONING_REPEATS: usize = 2048;
const LONG_TOOL_ARGUMENT_REPEATS: usize = 256;

/// The chat generation prompt for thinking mode ends inside the `think`
/// channel, so the output starts with reasoning.
const THINKING_PROMPT_TAIL: &str = "<|open|>think<|sep|>";

fn tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_special_token("<|open|>", 256)
        .with_special_token("<|close|>", 257)
        .with_special_token("<|sep|>", 258)
        .with_special_token("<|end_of_msg|>", 259)
}

fn argument(key: &str, arg_type: &str, value: &str) -> String {
    format!(
        "<|open|>argument key=\"{key}\" type=\"{arg_type}\"<|sep|>{value}<|close|>argument<|sep|>"
    )
}

fn call(tool: &str, index: usize, body: &str) -> String {
    format!("<|open|>call tool=\"{tool}\" index=\"{index}\"<|sep|>{body}<|close|>call<|sep|>")
}

fn message(reasoning: &str, response: &str, calls: &str) -> String {
    format!(
        "{reasoning}<|close|>think<|sep|>\
         <|open|>response<|sep|>{response}<|close|>response<|sep|>\
         <|open|>tools<|sep|>{calls}<|close|>tools<|sep|>\
         <|close|>message<|sep|>"
    )
}

fn mixed_fixture() -> String {
    let calls = [
        call(
            "convert",
            1,
            &[
                argument("whole", "number", "114.514"),
                argument("flag", "boolean", "true"),
                argument("name", "string", "demo"),
                argument("payload", "object", r#"{"count":42,"tags":["red","blue"]}"#),
            ]
            .concat(),
        ),
        call("update_record", 2, &argument("id", "integer", "7")),
    ]
    .concat();
    message(
        "I should inspect the data first.",
        "I will convert it.",
        &calls,
    )
}

fn long_reasoning_fixture() -> String {
    let line = "Ordinary reasoning text with no K3 channel markers at all.\n";
    message(&line.repeat(LONG_REASONING_REPEATS), "Done.", "")
}

fn long_tool_argument_fixture() -> String {
    let line = "<section><p>Literal <tag> and quotes \" inside content.</p></section>\n";
    let content = line.repeat(LONG_TOOL_ARGUMENT_REPEATS);
    let body = [
        argument("path", "string", "index.html"),
        argument("content", "string", &content),
    ]
    .concat();
    message(
        "Write the file.",
        "I will write the file.",
        &call("write_file", 1, &body),
    )
}

fn parser(tokenizer: &TestTokenizer) -> KimiK3UnifiedParser {
    KimiK3UnifiedParser::new(&[], Arc::new(tokenizer.clone()))
        .expect("Kimi K3 unified parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    text: &str,
    expected_normal_text: &str,
    expected_reasoning_len: usize,
    expected_calls_len: usize,
) {
    let tokenizer = tokenizer();
    let prompt_token_ids = tokenizer.encode(THINKING_PROMPT_TAIL, false).unwrap();
    let chunks = attributed_chunks(&tokenizer, text, CHUNK_CHARS);

    // Check the fixture once, outside the measurement: a marker lost to
    // missing attribution would silently turn the whole stream into content.
    let summary = feed_unified_parser(&mut parser(&tokenizer), &prompt_token_ids, chunks.clone());
    assert_eq!(summary.normal_text, expected_normal_text, "{name}");
    assert_eq!(
        summary.reasoning_text.len(),
        expected_reasoning_len,
        "{name}"
    );
    assert_eq!(summary.calls_len, expected_calls_len, "{name}");

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("reuse_parser", |b| {
        let mut parser = parser(&tokenizer);
        b.iter_batched(
            || chunks.clone(),
            |chunks| {
                black_box(feed_unified_parser(
                    &mut parser,
                    &prompt_token_ids,
                    black_box(chunks),
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("create_parser", |b| {
        b.iter_batched(
            || (parser(&tokenizer), chunks.clone()),
            |(mut parser, chunks)| {
                black_box(feed_unified_parser(
                    &mut parser,
                    &prompt_token_ids,
                    black_box(chunks),
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_kimi_k3(c: &mut Criterion) {
    run_stream_group(
        c,
        "kimi_k3/mixed_complex_tool_call",
        &mixed_fixture(),
        "I will convert it.",
        "I should inspect the data first.".len(),
        2,
    );

    let long_reasoning = long_reasoning_fixture();
    let reasoning_len = long_reasoning.find("<|close|>think").unwrap();
    run_stream_group(
        c,
        "kimi_k3/long_reasoning",
        &long_reasoning,
        "Done.",
        reasoning_len,
        0,
    );

    run_stream_group(
        c,
        "kimi_k3/long_tool_argument",
        &long_tool_argument_fixture(),
        "I will write the file.",
        "Write the file.".len(),
        1,
    );
}

criterion_group!(benches, bench_kimi_k3);
criterion_main!(benches);
