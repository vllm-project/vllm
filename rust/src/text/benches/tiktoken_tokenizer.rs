use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hf_hub::api::sync::ApiBuilder;
use vllm_text::tokenizer::{TiktokenTokenizer, Tokenizer};

const MODEL_ID: &str = "moonshotai/Kimi-K2.5";
const SAMPLE_TEXT: &str = "\
<think>
I'm sure it's fine, but I can't say I'd trust that it's what we'd ship.
</think>
请用中英混合总结以下需求，并保留 tool-call marker:
<|tool_calls_section_begin|>{\"name\":\"summarize\",\"arguments\":{\"style\":\"brief\"}}<|tool_calls_section_end|>
The service should stop cleanly at EOS, avoid leaking the next template turn, and keep decode latency low.
";

struct BenchFixture {
    riptoken: TiktokenTokenizer,
    tiktoken_rs: TiktokenTokenizer,
    text: String,
    token_ids: Vec<u32>,
}

impl BenchFixture {
    fn load() -> Self {
        let path = tiktoken_model();
        let riptoken = TiktokenTokenizer::new_riptoken(&path).expect("load riptoken tokenizer");
        let tiktoken_rs =
            TiktokenTokenizer::new_tiktoken_rs(&path).expect("load tiktoken-rs tokenizer");

        let text = SAMPLE_TEXT.repeat(32);
        let riptoken_token_ids =
            riptoken.encode(text.as_str(), false).expect("encode sample text with riptoken");
        let tiktoken_rs_token_ids = tiktoken_rs
            .encode(text.as_str(), false)
            .expect("encode sample text with tiktoken-rs");
        assert_eq!(riptoken_token_ids, tiktoken_rs_token_ids);

        let riptoken_decoded = riptoken
            .decode(riptoken_token_ids.as_slice(), false)
            .expect("decode sample token ids with riptoken");
        let tiktoken_rs_decoded = tiktoken_rs
            .decode(riptoken_token_ids.as_slice(), false)
            .expect("decode sample token ids with tiktoken-rs");
        assert_eq!(riptoken_decoded, tiktoken_rs_decoded);

        Self {
            riptoken,
            tiktoken_rs,
            text,
            token_ids: riptoken_token_ids,
        }
    }
}

fn tiktoken_model() -> std::path::PathBuf {
    let repo = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .expect("build hf-hub api")
        .model(MODEL_ID.to_string());
    repo.get("config.json").expect("fetch config.json from hf-hub");
    repo.get("tokenizer_config.json")
        .expect("fetch tokenizer_config.json from hf-hub");
    repo.get("tiktoken.model").expect("fetch tiktoken.model from hf-hub")
}

fn bench_encode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tiktoken_encode");
    group.throughput(Throughput::Bytes(fixture.text.len() as u64));

    group.bench_function("riptoken", |b| {
        b.iter(|| {
            fixture
                .riptoken
                .encode(black_box(fixture.text.as_str()), black_box(false))
                .expect("encode sample text with riptoken")
        })
    });
    group.bench_function("tiktoken_rs", |b| {
        b.iter(|| {
            fixture
                .tiktoken_rs
                .encode(black_box(fixture.text.as_str()), black_box(false))
                .expect("encode sample text with tiktoken-rs")
        })
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tiktoken_decode");
    group.throughput(Throughput::Elements(fixture.token_ids.len() as u64));

    group.bench_function("riptoken", |b| {
        b.iter(|| {
            fixture
                .riptoken
                .decode(black_box(fixture.token_ids.as_slice()), black_box(false))
                .expect("decode sample token ids with riptoken")
        })
    });
    group.bench_function("tiktoken_rs", |b| {
        b.iter(|| {
            fixture
                .tiktoken_rs
                .decode(black_box(fixture.token_ids.as_slice()), black_box(false))
                .expect("decode sample token ids with tiktoken-rs")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
