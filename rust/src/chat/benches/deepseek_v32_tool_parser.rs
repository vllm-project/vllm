use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_chat::test_utils::tool_parser::{split_by_chars, test_tools};
use vllm_chat::{ChatTool, ToolParser, ToolParserFactory};

mod utils;
use utils::feed_parser;

const PARSER_NAME: &str = "deepseek_v32";
const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<｜DSML｜function_calls>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">Hangzhou</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"date\" string=\"true\">2026-04-28</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "<｜DSML｜invoke name=\"get_weather\">\n",
        "<｜DSML｜parameter name=\"location\" string=\"true\">San Francisco</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"date\" string=\"true\">2026-04-28</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"unit\" string=\"true\">fahrenheit</｜DSML｜parameter>\n",
        "<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n",
        "</｜DSML｜invoke>\n",
        "</｜DSML｜function_calls>",
    )
    .to_string()
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no DSML tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn parser(tools: &[ChatTool]) -> Box<dyn ToolParser> {
    ToolParserFactory::global()
        .create(PARSER_NAME, tools)
        .expect("DeepSeek V3.2 parser should be registered")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[ChatTool],
    text: &str,
    chunk_chars: usize,
    expected_normal_text: &str,
    expected_calls_len: usize,
) {
    let chunks = split_by_chars(text, chunk_chars);

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

fn bench_deepseek_v32_tool_parser(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "deepseek_v32_tool_parser/mixed_text_tool_call",
        &tools,
        &mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "deepseek_v32_tool_parser/long_normal_text",
        &tools,
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_deepseek_v32_tool_parser);
criterion_main!(benches);
