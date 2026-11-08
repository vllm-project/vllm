// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tool_parser::parsers::PythonicParser as ExternalPythonicParser;
use vllm_parser::tool::test_utils::{split_by_chars, test_tools};
use vllm_parser::tool::{PythonicToolParser, Tool, ToolParser};

mod utils;
use utils::{feed_external_parser, feed_parser, openai_tools};

const CHUNK_CHARS: usize = 7;
const LONG_ARGUMENT_BYTES: usize = 64 * 1024;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn mixed_fixture() -> String {
    "[get_weather(city='Hangzhou', days=3), \
     convert(whole=42.5, flag=True, payload={'nested': ['x', None]}, items=[1, 2, 3], empty='')]"
        .to_string()
}

fn long_string_argument_fixture() -> String {
    format!("[convert(empty='{}')]", "x".repeat(LONG_ARGUMENT_BYTES))
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no pythonic tool call at the root.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn native_parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    PythonicToolParser::create(tools).expect("pythonic parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[Tool],
    text: &str,
    expected_normal_text: &str,
    expected_native_calls_len: usize,
) {
    let chunks = split_by_chars(text, CHUNK_CHARS);
    let openai_tools = openai_tools(tools);

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("native_reuse_parser", |b| {
        let mut parser = native_parser(tools);
        b.iter(|| {
            let result = feed_parser(&mut *parser, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_native_calls_len);
            black_box(result);
        })
    });

    group.bench_function("native_create_parser", |b| {
        b.iter_batched(
            || native_parser(tools),
            |mut parser| {
                let result = feed_parser(&mut *parser, black_box(&chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                debug_assert_eq!(result.1, expected_native_calls_len);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("external_reuse_parser", |b| {
        let mut parser = ExternalPythonicParser::new();
        b.iter(|| {
            let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
            black_box(result);
        })
    });

    group.bench_function("external_create_parser", |b| {
        b.iter_batched(
            ExternalPythonicParser::new,
            |mut parser| {
                let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_pythonic(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_string_argument = long_string_argument_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(c, "pythonic/tool_call_list", &tools, &mixed_text, "", 2);
    run_stream_group(
        c,
        "pythonic/long_string_argument",
        &tools,
        &long_string_argument,
        "",
        1,
    );
    run_stream_group(
        c,
        "pythonic/long_normal_text",
        &tools,
        &long_normal_text,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_pythonic);
criterion_main!(benches);
