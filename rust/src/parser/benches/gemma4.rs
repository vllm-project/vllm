// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::tool::Tool;
use vllm_parser::tool::test_utils::test_tools;
use vllm_parser::unified::Gemma4UnifiedParser;
use vllm_tokenizer::test_utils::TestTokenizer;

mod utils;
use utils::{attributed_chunks, feed_unified_parser};

const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;
const LONG_TOOL_ARGUMENT_REPEATS: usize = 256;

fn mixed_fixture() -> String {
    concat!(
        "I will inspect the data before answering.\n",
        "<|tool_call>",
        "call:convert{",
        "whole:114.514,",
        "flag:true,",
        "empty:<|\"|><|\"|>,",
        "payload:{",
        "name:<|\"|>demo<|\"|>,",
        "count:42,",
        "enabled:false,",
        "missing:null,",
        "nested:{level:2,label:<|\"|>deep<|\"|>},",
        "tags:[<|\"|>red<|\"|>,<|\"|>blue<|\"|>,3,true,null,{kind:<|\"|>leaf<|\"|>}]",
        "},",
        "items:[",
        "<|\"|>alpha<|\"|>,",
        "{key:<|\"|>value<|\"|>,score:0.75},",
        "[1,2,3]",
        "]",
        "}",
        "<tool_call|>",
        "<|tool_call>",
        "call:update_record{",
        "data:{id:7,active:true,notes:[<|\"|>keep<|\"|>,<|\"|>review<|\"|>]}",
        "}",
        "<tool_call|>",
        " Finished.",
    )
    .to_string()
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no Gemma4 tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn long_tool_argument_fixture() -> String {
    let line =
        "<section><p>Literal } and <tool_call|> marker-shaped text inside content.</p></section>\n";
    format!(
        concat!(
            "I will write the file.\n",
            "<|tool_call>",
            "call:write_file{{",
            "path:<|\"|>index.html<|\"|>,",
            "content:<|\"|>{}<|\"|>",
            "}}",
            "<tool_call|>",
            "Done."
        ),
        line.repeat(LONG_TOOL_ARGUMENT_REPEATS)
    )
}

fn tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_special_token("<|tool_call>", 256)
        .with_special_token("<tool_call|>", 257)
        .with_special_token("<|\"|>", 258)
        .with_special_token("<|channel>", 259)
        .with_special_token("<channel|>", 260)
}

fn parser(tools: &[Tool], tokenizer: &TestTokenizer) -> Gemma4UnifiedParser {
    Gemma4UnifiedParser::new(tools, Arc::new(tokenizer.clone()))
        .expect("Gemma4 unified parser should initialize")
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
    // missing attribution would silently turn the whole stream into content.
    let summary = feed_unified_parser(&mut parser(tools, &tokenizer), &[], chunks.clone());
    assert_eq!(summary.normal_text, expected_normal_text, "{name}");
    assert_eq!(summary.calls_len, expected_calls_len, "{name}");

    let mut group = c.benchmark_group(name);
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("reuse_parser", |b| {
        let mut parser = parser(tools, &tokenizer);
        b.iter_batched(
            || chunks.clone(),
            |chunks| black_box(feed_unified_parser(&mut parser, &[], black_box(chunks))),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("create_parser", |b| {
        b.iter_batched(
            || (parser(tools, &tokenizer), chunks.clone()),
            |(mut parser, chunks)| {
                black_box(feed_unified_parser(&mut parser, &[], black_box(chunks)))
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_gemma4(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_tool_argument = long_tool_argument_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "gemma4/mixed_complex_tool_call",
        &tools,
        &mixed_text,
        CHUNK_CHARS,
        "I will inspect the data before answering.\n Finished.",
        2,
    );

    run_stream_group(
        c,
        "gemma4/long_tool_argument",
        &tools,
        &long_tool_argument,
        CHUNK_CHARS,
        "I will write the file.\nDone.",
        1,
    );

    run_stream_group(
        c,
        "gemma4/long_normal_text",
        &tools,
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_gemma4);
criterion_main!(benches);
