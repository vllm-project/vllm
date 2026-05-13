use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tool_parser::parsers::LlamaParser as ExternalLlamaParser;
use vllm_chat::test_utils::tool_parser::{split_by_chars, test_tools};
use vllm_chat::{ChatTool, ToolParser, ToolParserFactory};

mod utils;
use utils::{feed_external_parser, feed_parser, openai_tools};

const PARSER_NAME: &str = "llama3_json";
const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_CHUNK_CHARS: usize = 37;
const LONG_NORMAL_TEXT_REPEATS: usize = 4096;

fn tool_call(function_name: &str, parameters: &str) -> String {
    format!(r#"{{"name":"{function_name}","parameters":{parameters}}}"#)
}

fn mixed_fixture() -> String {
    format!(
        "{}; {}",
        tool_call("get_weather", r#"{"location":"Hangzhou","days":3}"#),
        tool_call(
            "convert",
            r#"{"whole":42.5,"flag":true,"payload":{"nested":["x",null]},"items":[1,2,3],"empty":""}"#
        ),
    )
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no Llama JSON tool call at the root.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn native_parser(tools: &[ChatTool]) -> Box<dyn ToolParser> {
    ToolParserFactory::global()
        .create(PARSER_NAME, tools)
        .expect("Llama JSON parser should be registered")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[ChatTool],
    text: &str,
    chunk_chars: usize,
    expected_normal_text: &str,
    expected_native_calls_len: usize,
) {
    let chunks = split_by_chars(text, chunk_chars);
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
        let mut parser = ExternalLlamaParser::new();
        b.iter(|| {
            let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
            black_box(result);
        })
    });

    group.bench_function("external_create_parser", |b| {
        b.iter_batched(
            ExternalLlamaParser::new,
            |mut parser| {
                let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_llama3_json_tool_parser(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "llama3_json_tool_parser/mixed_text_tool_call",
        &tools,
        &mixed_text,
        CHUNK_CHARS,
        "",
        2,
    );

    run_stream_group(
        c,
        "llama3_json_tool_parser/long_normal_text",
        &tools,
        &long_normal_text,
        LONG_NORMAL_TEXT_CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_llama3_json_tool_parser);
criterion_main!(benches);
