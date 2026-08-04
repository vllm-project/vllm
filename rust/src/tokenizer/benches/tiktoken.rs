// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hf_hub::api::tokio::ApiBuilder;
use tokio::runtime::Runtime;
use vllm_tokenizer::{TiktokenTokenizer, Tokenizer};

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
    fastokens: TiktokenTokenizer,
    riptoken: TiktokenTokenizer,
    tiktoken_rs: TiktokenTokenizer,
    text: String,
    token_ids: Vec<u32>,
}

impl BenchFixture {
    fn load() -> Self {
        let path = tiktoken_model();
        let fastokens = TiktokenTokenizer::new_fastokens(&path).expect("load fastokens tokenizer");
        let riptoken = TiktokenTokenizer::new_riptoken(&path).expect("load riptoken tokenizer");
        let tiktoken_rs =
            TiktokenTokenizer::new_tiktoken_rs(&path).expect("load tiktoken-rs tokenizer");

        let text = SAMPLE_TEXT.repeat(32);
        let fastokens_token_ids = fastokens
            .encode(text.as_str(), false)
            .expect("encode sample text with fastokens");
        let riptoken_token_ids =
            riptoken.encode(text.as_str(), false).expect("encode sample text with riptoken");
        let tiktoken_rs_token_ids = tiktoken_rs
            .encode(text.as_str(), false)
            .expect("encode sample text with tiktoken-rs");
        assert_eq!(fastokens_token_ids, riptoken_token_ids);
        assert_eq!(fastokens_token_ids, tiktoken_rs_token_ids);

        let fastokens_ordinary_token_ids = fastokens
            .encode_ordinary(text.as_str())
            .expect("ordinary-encode sample text with fastokens");
        let riptoken_ordinary_token_ids = riptoken
            .encode_ordinary(text.as_str())
            .expect("ordinary-encode sample text with riptoken");
        let tiktoken_rs_ordinary_token_ids = tiktoken_rs
            .encode_ordinary(text.as_str())
            .expect("ordinary-encode sample text with tiktoken-rs");
        assert_eq!(fastokens_ordinary_token_ids, riptoken_ordinary_token_ids);
        assert_eq!(fastokens_ordinary_token_ids, tiktoken_rs_ordinary_token_ids);

        let fastokens_decoded = fastokens
            .decode(fastokens_token_ids.as_slice(), false)
            .expect("decode sample token ids with fastokens");
        let riptoken_decoded = riptoken
            .decode(fastokens_token_ids.as_slice(), false)
            .expect("decode sample token ids with riptoken");
        let tiktoken_rs_decoded = tiktoken_rs
            .decode(fastokens_token_ids.as_slice(), false)
            .expect("decode sample token ids with tiktoken-rs");
        assert_eq!(fastokens_decoded, riptoken_decoded);
        assert_eq!(fastokens_decoded, tiktoken_rs_decoded);

        Self {
            fastokens,
            riptoken,
            tiktoken_rs,
            text,
            token_ids: fastokens_token_ids,
        }
    }
}

fn tiktoken_model() -> std::path::PathBuf {
    Runtime::new().expect("build tokio runtime").block_on(async {
        let repo = ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .expect("build hf-hub api")
            .model(MODEL_ID.to_string());
        repo.get("config.json").await.expect("fetch config.json from hf-hub");
        repo.get("tokenizer_config.json")
            .await
            .expect("fetch tokenizer_config.json from hf-hub");
        repo.get("tiktoken.model").await.expect("fetch tiktoken.model from hf-hub")
    })
}

fn bench_encode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tiktoken_encode");
    group.throughput(Throughput::Bytes(fixture.text.len() as u64));

    group.bench_function("fastokens", |b| {
        b.iter(|| {
            fixture
                .fastokens
                .encode(black_box(fixture.text.as_str()), black_box(false))
                .expect("encode sample text with fastokens")
        })
    });
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

fn bench_encode_ordinary(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tiktoken_encode_ordinary");
    group.throughput(Throughput::Bytes(fixture.text.len() as u64));

    group.bench_function("fastokens", |b| {
        b.iter(|| {
            fixture
                .fastokens
                .encode_ordinary(black_box(fixture.text.as_str()))
                .expect("ordinary-encode sample text with fastokens")
        })
    });
    group.bench_function("riptoken", |b| {
        b.iter(|| {
            fixture
                .riptoken
                .encode_ordinary(black_box(fixture.text.as_str()))
                .expect("ordinary-encode sample text with riptoken")
        })
    });
    group.bench_function("tiktoken_rs", |b| {
        b.iter(|| {
            fixture
                .tiktoken_rs
                .encode_ordinary(black_box(fixture.text.as_str()))
                .expect("ordinary-encode sample text with tiktoken-rs")
        })
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tiktoken_decode");
    group.throughput(Throughput::Elements(fixture.token_ids.len() as u64));

    group.bench_function("fastokens", |b| {
        b.iter(|| {
            fixture
                .fastokens
                .decode(black_box(fixture.token_ids.as_slice()), black_box(false))
                .expect("decode sample token ids with fastokens")
        })
    });
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

criterion_group!(benches, bench_encode, bench_encode_ordinary, bench_decode);
criterion_main!(benches);
