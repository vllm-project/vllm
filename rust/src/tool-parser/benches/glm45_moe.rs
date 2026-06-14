use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tool_parser::parsers::Glm4MoeParser as ExternalGlm4MoeParser;
use vllm_tool_parser::test_utils::{split_by_chars, test_tools};
use vllm_tool_parser::{Glm45MoeToolParser, Glm47MoeToolParser, Tool, ToolParser};

mod utils;
use utils::{feed_external_parser, feed_parser, openai_tools};

const GLM45_PARSER_NAME: &str = "glm45";
const GLM47_PARSER_NAME: &str = "glm47";
const CHUNK_CHARS: usize = 7;
const LONG_NORMAL_TEXT_REPEATS: usize = 2048;

fn glm45_mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>\n",
        "<arg_value>Hangzhou</arg_value>\n",
        "<arg_key>date</arg_key>\n",
        "<arg_value>2026-05-07</arg_value>\n",
        "<arg_key>unit</arg_key>\n",
        "<arg_value>celsius</arg_value>\n",
        "<arg_key>days</arg_key>\n",
        "<arg_value>3</arg_value>\n",
        "</tool_call>\n",
        "<tool_call>get_weather\n",
        "<arg_key>city</arg_key>\n",
        "<arg_value>San Francisco</arg_value>\n",
        "<arg_key>date</arg_key>\n",
        "<arg_value>2026-05-07</arg_value>\n",
        "<arg_key>unit</arg_key>\n",
        "<arg_value>fahrenheit</arg_value>\n",
        "<arg_key>days</arg_key>\n",
        "<arg_value>2</arg_value>\n",
        "</tool_call>",
    )
    .to_string()
}

fn glm47_mixed_fixture() -> String {
    concat!(
        "I will check two cities before answering.\n",
        "<tool_call>get_weather",
        "<arg_key>city</arg_key>",
        "<arg_value>Hangzhou</arg_value>",
        "<arg_key>date</arg_key>",
        "<arg_value>2026-05-07</arg_value>",
        "<arg_key>unit</arg_key>",
        "<arg_value>celsius</arg_value>",
        "<arg_key>days</arg_key>",
        "<arg_value>3</arg_value>",
        "</tool_call>",
        "<tool_call>get_weather",
        "<arg_key>city</arg_key>",
        "<arg_value>San Francisco</arg_value>",
        "<arg_key>date</arg_key>",
        "<arg_value>2026-05-07</arg_value>",
        "<arg_key>unit</arg_key>",
        "<arg_value>fahrenheit</arg_value>",
        "<arg_key>days</arg_key>",
        "<arg_value>2</arg_value>",
        "</tool_call>",
    )
    .to_string()
}

fn long_normal_text_fixture() -> String {
    let line = "This is ordinary assistant text with no GLM MoE tool markers at all.\n";
    line.repeat(LONG_NORMAL_TEXT_REPEATS)
}

fn native_parser(name: &str, tools: &[Tool]) -> Box<dyn ToolParser> {
    match name {
        GLM45_PARSER_NAME => Glm45MoeToolParser::create(tools),
        GLM47_PARSER_NAME => Glm47MoeToolParser::create(tools),
        _ => unreachable!("unexpected GLM parser name"),
    }
    .expect("GLM MoE parser should initialize")
}

fn external_parser(name: &str) -> ExternalGlm4MoeParser {
    match name {
        GLM45_PARSER_NAME => ExternalGlm4MoeParser::glm45(),
        GLM47_PARSER_NAME => ExternalGlm4MoeParser::glm47(),
        _ => unreachable!("unexpected GLM parser name"),
    }
}

fn run_stream_group(
    c: &mut Criterion,
    name: &str,
    parser_name: &str,
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
        let mut parser = native_parser(parser_name, tools);
        b.iter(|| {
            let result = feed_parser(&mut *parser, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            debug_assert_eq!(result.1, expected_native_calls_len);
            black_box(result);
        })
    });

    group.bench_function("native_create_parser", |b| {
        b.iter_batched(
            || native_parser(parser_name, tools),
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
        let mut parser = external_parser(parser_name);
        b.iter(|| {
            let result = feed_external_parser(&mut parser, &openai_tools, black_box(&chunks));
            debug_assert_eq!(result.0, expected_normal_text);
            black_box(result);
        })
    });

    group.bench_function("external_create_parser", |b| {
        b.iter_batched(
            || external_parser(parser_name),
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

fn bench_glm45_moe(c: &mut Criterion) {
    let tools = test_tools();
    let glm45_mixed_text = glm45_mixed_fixture();
    let glm47_mixed_text = glm47_mixed_fixture();
    let long_normal_text = long_normal_text_fixture();

    run_stream_group(
        c,
        "glm45/mixed_text_tool_call",
        GLM45_PARSER_NAME,
        &tools,
        &glm45_mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "glm47/mixed_text_tool_call",
        GLM47_PARSER_NAME,
        &tools,
        &glm47_mixed_text,
        CHUNK_CHARS,
        "I will check two cities before answering.\n",
        2,
    );

    run_stream_group(
        c,
        "glm45/long_normal_text",
        GLM45_PARSER_NAME,
        &tools,
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );

    run_stream_group(
        c,
        "glm47/long_normal_text",
        GLM47_PARSER_NAME,
        &tools,
        &long_normal_text,
        CHUNK_CHARS,
        &long_normal_text,
        0,
    );
}

criterion_group!(benches, bench_glm45_moe);
criterion_main!(benches);
