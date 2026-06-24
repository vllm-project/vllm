use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use vllm_parser::tool::Tool;
use vllm_parser::tool::test_utils::{split_by_chars, test_tools};
use vllm_parser::unified::{Gemma4UnifiedParser, UnifiedParser, UnifiedParserEvent};
use vllm_tokenizer::Tokenizer;

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

struct BenchTokenizer;

impl Tokenizer for BenchTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_tokenizer::Result<String> {
        Ok(token_ids
            .iter()
            .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|channel>" => Some(100),
            "<channel|>" => Some(101),
            "<|tool_call>" => Some(102),
            "<tool_call|>" => Some(103),
            "<|turn>" => Some(104),
            "<|tool_response>" => Some(105),
            _ => None,
        }
    }
}

fn parser(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(Gemma4UnifiedParser::new(tools, Arc::new(BenchTokenizer)).unwrap())
}

fn feed_parser(parser: &mut dyn UnifiedParser, chunks: &[&str]) -> (String, usize) {
    let mut normal_text = String::new();
    let mut calls_len = 0;
    for chunk in chunks {
        let mut output = vllm_parser::unified::UnifiedParserOutput::default();
        parser.parse_into(chunk, &mut output).unwrap();
        collect_output(output, &mut normal_text, &mut calls_len);
    }
    collect_output(parser.finish().unwrap(), &mut normal_text, &mut calls_len);
    (normal_text, calls_len)
}

fn collect_output(
    output: vllm_parser::unified::UnifiedParserOutput,
    normal_text: &mut String,
    calls_len: &mut usize,
) {
    for event in output.events {
        match event {
            UnifiedParserEvent::Text(text) => normal_text.push_str(&text),
            UnifiedParserEvent::Reasoning(_) => {}
            UnifiedParserEvent::ToolCall(_) => *calls_len += 1,
        }
    }
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
