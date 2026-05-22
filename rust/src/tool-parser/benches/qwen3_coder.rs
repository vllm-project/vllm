use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tool_parser::parsers::QwenCoderParser as ExternalQwenCoderParser;
use vllm_tool_parser::test_utils::{split_by_chars, test_tools};
use vllm_tool_parser::{Qwen3CoderToolParser, Tool, ToolParser};

mod utils;
use utils::{feed_external_parser, feed_parser, openai_tools};

const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<tool_call>\n",
        "<function=get_weather>\n",
        "<parameter=location>Hangzhou</parameter>\n",
        "<parameter=date>2026-04-29</parameter>\n",
        "<parameter=unit>celsius</parameter>\n",
        "<parameter=days>3</parameter>\n",
        "</function>\n",
        "</tool_call>\n",
        "<tool_call>\n",
        "<function=get_weather>\n",
        "<parameter=location>San Francisco</parameter>\n",
        "<parameter=date>2026-04-29</parameter>\n",
        "<parameter=unit>fahrenheit</parameter>\n",
        "<parameter=days>2</parameter>\n",
        "</function>\n",
        "</tool_call>",
    )
    .to_string()
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no Qwen Coder tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn native_parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    Qwen3CoderToolParser::create(tools).expect("Qwen Coder parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[Tool],
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
        let mut parser = ExternalQwenCoderParser::new();
        b.iter(|| {
            let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            black_box(result);
        })
    });

    group.bench_function("external_create_parser", |b| {
        b.iter_batched(
            ExternalQwenCoderParser::new,
            |mut parser| {
                let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
                debug_assert_eq!(result.0, expected_normal_text);
                black_box(result);
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_qwen3_coder(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "qwen3_coder/mixed_text_tool_call",
        &tools,
        &mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "qwen3_coder/long_normal_text",
        &tools,
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_qwen3_coder);
criterion_main!(benches);
