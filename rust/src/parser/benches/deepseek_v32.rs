// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::tool::test_utils::test_tools;
use vllm_parser::tool::{DeepSeekV32ToolParser, Tool, ToolParser};
use vllm_tokenizer::test_utils::TestTokenizer;

mod utils;
use utils::{attributed_chunks, feed_attributed_parser};

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

/// DeepSeek V3.2 has `｜DSML｜` as a special token, so the parser guards its
/// markers with it.
fn tokenizer() -> TestTokenizer {
    TestTokenizer::new().with_special_token("｜DSML｜", 256)
}

fn parser(tools: &[Tool], tokenizer: &TestTokenizer) -> Box<dyn ToolParser> {
    DeepSeekV32ToolParser::create(tools, Arc::new(tokenizer.clone()))
        .expect("DeepSeek V3.2 parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    tools: &[Tool],
    text: &str,
    chunk_chars: usize,
    expected_normal_text: &str,
    expected_calls_len: usize,
) {
    let tokenizer = tokenizer();
    let chunks = attributed_chunks(&tokenizer, text, chunk_chars);

    // Check the fixture once, outside the measurement: a marker lost to
    // missing attribution would silently turn every call into text.
    let result = feed_attributed_parser(&mut *parser(tools, &tokenizer), chunks.clone());
    assert_eq!(result.0, expected_normal_text, "{name}");
    assert_eq!(result.1, expected_calls_len, "{name}");

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("reuse_parser", |b| {
        let mut parser = parser(tools, &tokenizer);
        b.iter_batched(
            || chunks.clone(),
            |chunks| black_box(feed_attributed_parser(&mut *parser, black_box(chunks))),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("create_parser", |b| {
        b.iter_batched(
            || (parser(tools, &tokenizer), chunks.clone()),
            |(mut parser, chunks)| {
                black_box(feed_attributed_parser(&mut *parser, black_box(chunks)))
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_deepseek_v32(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "deepseek_v32/mixed_text_tool_call",
        &tools,
        &mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "deepseek_v32/long_normal_text",
        &tools,
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_deepseek_v32);
criterion_main!(benches);
