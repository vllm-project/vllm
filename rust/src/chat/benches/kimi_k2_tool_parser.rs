use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tool_parser::parsers::KimiK2Parser as ExternalKimiK2Parser;
use vllm_chat::test_utils::tool_parser::{split_by_chars, test_tools};
use vllm_chat::{ChatTool, ToolParser, ToolParserFactory};

mod utils;
use utils::{feed_external_parser, feed_parser, openai_tools};

const PARSER_NAME: &str = "kimi_k2";
const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>{\"location\":\"Hangzhou\",\"days\":3}",
        "<|tool_call_end|>",
        "<|tool_call_begin|>functions.get_weather:1",
        "<|tool_call_argument_begin|>{\"location\":\"San Francisco\",\"days\":2}",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
    )
    .to_string()
}

fn mixed_chunks() -> Vec<&'static str> {
    vec![
        "I will check two cities before answering.\n",
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>functions.get_weather:0",
        "<|tool_call_argument_begin|>",
        "{\"location\":",
        "\"Hangzhou\",",
        "\"days\":3}",
        "<|tool_call_end|>",
        "<|tool_call_begin|>functions.get_weather:1",
        "<|tool_call_argument_begin|>",
        "{\"location\":",
        "\"San Francisco\",",
        "\"days\":2}",
        "<|tool_call_end|>",
        "<|tool_calls_section_end|>",
    ]
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no Kimi K2 tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn native_parser(tools: &[ChatTool]) -> Box<dyn ToolParser> {
    ToolParserFactory::global()
        .create(PARSER_NAME, tools)
        .expect("Kimi K2 parser should be registered")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[ChatTool],
    text: &str,
    chunks: &[&str],
    expected_normal_text: &str,
    expected_native_calls_len: usize,
) {
    let openai_tools = openai_tools(tools);

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("native_reuse_parser", |b| {
        let mut parser = native_parser(tools);
        b.iter(|| {
            let result = feed_parser(&mut *parser, black_box(chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_native_calls_len);
            black_box(result);
        })
    });

    group.bench_function("native_create_parser", |b| {
        b.iter_batched(
            || native_parser(tools),
            |mut parser| {
                let result = feed_parser(&mut *parser, black_box(chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                debug_assert_eq!(result.1, expected_native_calls_len);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("external_reuse_parser", |b| {
        let mut parser = ExternalKimiK2Parser::new();
        b.iter(|| {
            let result = feed_external_parser(&mut parser, &openai_tools, black_box(chunks));
            black_box(result);
        })
    });

    group.bench_function("external_create_parser", |b| {
        b.iter_batched(
            ExternalKimiK2Parser::new,
            |mut parser| {
                let result = feed_external_parser(&mut parser, &openai_tools, black_box(chunks));
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_kimi_k2_tool_parser(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let mixed_chunks = mixed_chunks();
    let long_normal_text = long_normal_text_fixture();
    let long_normal_chunks = split_by_chars(&long_normal_text, CHUNK_CHARS);

    run_stream_group(
        c,
        "kimi_k2_tool_parser/mixed_text_tool_call",
        &tools,
        &mixed_text,
        &mixed_chunks,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "kimi_k2_tool_parser/long_normal_text",
        &tools,
        &long_normal_text,
        &long_normal_chunks,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_kimi_k2_tool_parser);
criterion_main!(benches);
