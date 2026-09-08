// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::tool::test_utils::{split_by_chars, test_tools};
use vllm_parser::tool::{Olmo3PythonicToolParser, Tool, ToolParser};

mod utils;
use utils::feed_parser;

const CHUNK_CHARS: usize = 7;
const LONG_ARGUMENT_BYTES: usize = 64 * 1024;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn mixed_fixture() -> String {
    concat!(
        "<function_calls>\n",
        "get_weather(city='Hangzhou', days=3)\n",
        "convert(whole=42.5, flag=true, payload={'nested': ['x', null]}, items=[1, 2, 3], empty='')\n",
        "</function_calls>"
    )
    .to_string()
}

fn long_string_argument_fixture() -> String {
    format!(
        "<function_calls>\nconvert(empty='{}')\n</function_calls>",
        "x".repeat(LONG_ARGUMENT_BYTES)
    )
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no OLMo 3 tool call block.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    Olmo3PythonicToolParser::create(tools).expect("OLMo 3 parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[Tool],
    text: &str,
    expected_normal_text: &str,
    expected_calls_len: usize,
) {
    let chunks = split_by_chars(text, CHUNK_CHARS);

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("reuse_parser", |b| {
        let mut parser = parser(tools);
        b.iter(|| {
            let result = feed_parser(&mut *parser, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_calls_len);
            black_box(result);
        })
    });

    group.bench_function("create_parser", |b| {
        b.iter_batched(
            || parser(tools),
            |mut parser| {
                let result = feed_parser(&mut *parser, black_box(&chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                debug_assert_eq!(result.1, expected_calls_len);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_olmo3(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_string_argument = long_string_argument_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(c, "olmo3/tool_call_block", &tools, &mixed_text, "", 2);
    run_stream_group(
        c,
        "olmo3/long_string_argument",
        &tools,
        &long_string_argument,
        "",
        1,
    );
    run_stream_group(
        c,
        "olmo3/long_normal_text",
        &tools,
        &long_normal_text,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_olmo3);
criterion_main!(benches);
