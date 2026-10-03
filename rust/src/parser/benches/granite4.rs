// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::tool::test_utils::{split_by_chars, test_tools};
use vllm_parser::tool::{Granite4ToolParser, Tool, ToolParser};

mod utils;
use utils::feed_parser;

const CHUNK_CHARS: usize = 7;
const LONG_ARGUMENT_BYTES: usize = 64 * 1024;

fn string_args_fixture() -> String {
    let arguments = format!(r#"{{"data":"{}"}}"#, "x".repeat(LONG_ARGUMENT_BYTES));
    let encoded_arguments = serde_json::to_string(&arguments).unwrap();
    format!(r#"<tool_call>{{"name":"f","arguments":{encoded_arguments}}}</tool_call>"#)
}

fn object_args_fixture() -> String {
    format!(
        r#"<tool_call>{{"name":"f","arguments":{{"data":"{}"}}}}</tool_call>"#,
        "x".repeat(LONG_ARGUMENT_BYTES)
    )
}

fn parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    Granite4ToolParser::create(tools).expect("Granite4 parser should initialize")
}

fn run_stream_group(c: &mut Criterion, name: &str, tools: &[Tool], text: &str) {
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
            debug_assert_eq!(result.0, "");
            debug_assert_eq!(result.1, 1);
            black_box(result);
        })
    });

    group.bench_function("create_parser", |b| {
        b.iter_batched(
            || parser(tools),
            |mut parser| {
                let result = feed_parser(&mut *parser, black_box(&chunks));
                debug_assert_eq!(result.0, "");
                debug_assert_eq!(result.1, 1);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_granite4(c: &mut Criterion) {
    let tools = test_tools();
    let string_args = string_args_fixture();
    let object_args = object_args_fixture();

    run_stream_group(c, "granite4/long_string_arguments", &tools, &string_args);
    run_stream_group(c, "granite4/long_object_arguments", &tools, &object_args);
}

criterion_group!(benches, bench_granite4);
criterion_main!(benches);
