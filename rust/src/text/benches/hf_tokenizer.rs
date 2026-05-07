use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hf_hub::api::sync::ApiBuilder;
use vllm_text::tokenizer::{HuggingFaceTokenizer, Tokenizer};

const MODEL_ID: &str = "Qwen/Qwen3.5-0.8B";
const SAMPLE_TEXT: &str = "\
<|im_start|>system
You are Qwen3.5, a helpful assistant.
<|im_end|>
<|im_start|>user
请用中英混合总结以下需求，并给出一个简短的 JSON 示例。
The service should stop cleanly at EOS, avoid leaking the next template turn, and keep decode latency low.
Input: 4 concurrent requests, 10240 prompt tokens, 16 generated tokens.
<|im_end|>
<|im_start|>assistant
";

struct BenchFixture {
    fastokens: HuggingFaceTokenizer,
    hf: HuggingFaceTokenizer,
    text: String,
    token_ids: Vec<u32>,
}

impl BenchFixture {
    fn load() -> Self {
        let path = tokenizer_json();
        let fastokens =
            HuggingFaceTokenizer::new_fastokens(&path).expect("load fastokens tokenizer");
        let hf = HuggingFaceTokenizer::new_hf(&path).expect("load huggingface tokenizer");

        let text = SAMPLE_TEXT.repeat(32);
        let hf_token_ids =
            hf.encode(text.as_str(), false).expect("encode sample text with hf tokenizer");
        let fastokens_token_ids = fastokens
            .encode(text.as_str(), false)
            .expect("encode sample text with fastokens");
        assert_eq!(fastokens_token_ids, hf_token_ids);

        let hf_decoded = hf
            .decode(hf_token_ids.as_slice(), false)
            .expect("decode sample token ids with hf tokenizer");
        let fastokens_decoded = fastokens
            .decode(hf_token_ids.as_slice(), false)
            .expect("decode sample token ids with fastokens");
        assert_eq!(fastokens_decoded, hf_decoded);

        Self {
            fastokens,
            hf,
            text,
            token_ids: hf_token_ids,
        }
    }
}

fn tokenizer_json() -> std::path::PathBuf {
    ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .expect("build hf-hub api")
        .model(MODEL_ID.to_string())
        .get("tokenizer.json")
        .expect("fetch tokenizer.json from hf-hub")
}

fn bench_encode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tokenizer_encode");
    group.throughput(Throughput::Bytes(fixture.text.len() as u64));

    group.bench_function("fastokens", |b| {
        b.iter(|| {
            fixture
                .fastokens
                .encode(black_box(fixture.text.as_str()), black_box(false))
                .expect("encode sample text with fastokens")
        })
    });
    group.bench_function("hf_tokenizers", |b| {
        b.iter(|| {
            fixture
                .hf
                .encode(black_box(fixture.text.as_str()), black_box(false))
                .expect("encode sample text with hf tokenizer")
        })
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let fixture = BenchFixture::load();
    let mut group = c.benchmark_group("tokenizer_decode");
    group.throughput(Throughput::Elements(fixture.token_ids.len() as u64));

    group.bench_function("fastokens", |b| {
        b.iter(|| {
            fixture
                .fastokens
                .decode(black_box(fixture.token_ids.as_slice()), black_box(false))
                .expect("decode sample token ids with fastokens")
        })
    });
    group.bench_function("hf_tokenizers", |b| {
        b.iter(|| {
            fixture
                .hf
                .decode(black_box(fixture.token_ids.as_slice()), black_box(false))
                .expect("decode sample token ids with hf tokenizer")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
