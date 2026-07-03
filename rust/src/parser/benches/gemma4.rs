use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::tool::test_utils::{split_by_chars, test_tools};
use vllm_parser::tool::{Tool, ToolParser};
use vllm_parser::unified::{Gemma4ConfigDrivenParser, Gemma4UnifiedParser};

mod utils;
use utils::{UnifiedToolParserAdapter, feed_parser};

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

type MakeParser = fn(&[Tool]) -> Box<dyn ToolParser>;

fn parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    UnifiedToolParserAdapter::<Gemma4UnifiedParser>::create(tools)
        .expect("Gemma4 unified parser should initialize")
}

fn config_driven_parser(tools: &[Tool]) -> Box<dyn ToolParser> {
    UnifiedToolParserAdapter::<Gemma4ConfigDrivenParser>::create(tools)
        .expect("Gemma4 config-driven parser should initialize")
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    make_parser: MakeParser,
    tools: &[Tool],
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
        let mut parser = make_parser(tools);
        b.iter(|| {
            let result = feed_parser(&mut *parser, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_calls_len);
            black_box(result);
        })
    });

    group.bench_function("create_parser", |b| {
        b.iter_batched(
            || make_parser(tools),
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

fn bench_gemma4(c: &mut Criterion) {
    let tools = test_tools();
    let mixed_text = mixed_fixture();
    let long_tool_argument = long_tool_argument_fixture();
    let long_normal_text = long_normal_text_fixture();

    let variants: [(&str, MakeParser); 2] = [
        ("gemma4", parser),
        ("gemma4_config_driven", config_driven_parser),
    ];

    for (prefix, make_parser) in variants {
        run_stream_group(
            c,
            &format!("{prefix}/mixed_complex_tool_call"),
            make_parser,
            &tools,
            &mixed_text,
            CHUNK_CHARS,
            "I will inspect the data before answering.\n Finished.",
            2,
        );

        run_stream_group(
            c,
            &format!("{prefix}/long_tool_argument"),
            make_parser,
            &tools,
            &long_tool_argument,
            CHUNK_CHARS,
            "I will write the file.\nDone.",
            1,
        );

        run_stream_group(
            c,
            &format!("{prefix}/long_normal_text"),
            make_parser,
            &tools,
            &long_normal_text,
            CHUNK_CHARS,
            &long_normal_text,
            0,
        );
    }
}

criterion_group!(benches, bench_gemma4);
criterion_main!(benches);
