use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};

const PLACEHOLDER_TOKEN_ID: u32 = u32::MAX - 1;
const EMBED_TOKEN_ID: u32 = u32::MAX - 2;

#[derive(Clone)]
struct Replacement {
    tokens: Vec<u32>,
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    prompt: Vec<u32>,
    replacements: Vec<Replacement>,
    final_token_count: usize,
}

fn make_case(
    name: &'static str,
    prompt_len: usize,
    placeholders: usize,
    replacement_len: usize,
) -> Case {
    let mut prompt = Vec::with_capacity(prompt_len);
    let spacing = prompt_len / (placeholders + 1);
    let mut next_placeholder = spacing;
    let mut inserted = 0;

    for index in 0..prompt_len {
        if inserted < placeholders && index == next_placeholder {
            prompt.push(PLACEHOLDER_TOKEN_ID);
            inserted += 1;
            next_placeholder += spacing;
        } else {
            prompt.push((index as u32 % 50_000) + 1);
        }
    }

    let replacement = Replacement {
        tokens: (0..replacement_len)
            .map(|index| {
                if index % 4 == 0 {
                    EMBED_TOKEN_ID
                } else {
                    (100_000 + index) as u32
                }
            })
            .collect(),
    };
    let replacements = vec![replacement; placeholders];
    let final_token_count = prompt_len - placeholders + placeholders * replacement_len;

    Case {
        name,
        prompt,
        replacements,
        final_token_count,
    }
}

fn find_next_token(haystack: &[u32], needle: u32, start: usize) -> Option<usize> {
    haystack
        .get(start..)?
        .iter()
        .position(|token| *token == needle)
        .map(|offset| start + offset)
}

fn expand_with_splice(prompt_token_ids: &mut Vec<u32>, replacements: &[Replacement]) -> Vec<usize> {
    let mut cursor = 0;
    let mut offsets = Vec::with_capacity(replacements.len());

    for replacement in replacements {
        let offset = find_next_token(prompt_token_ids, PLACEHOLDER_TOKEN_ID, cursor).unwrap();
        let replacement_tokens = replacement.tokens.clone();
        let is_embed = replacement_tokens
            .iter()
            .map(|token| *token == EMBED_TOKEN_ID)
            .collect::<Vec<_>>();
        black_box(is_embed);

        prompt_token_ids.splice(offset..offset + 1, replacement_tokens);
        offsets.push(offset);
        cursor = offset + replacement.tokens.len();
    }

    offsets
}

fn expand_single_pass(prompt_token_ids: &mut Vec<u32>, replacements: &[Replacement]) -> Vec<usize> {
    let total_replacement_len =
        replacements.iter().map(|replacement| replacement.tokens.len()).sum::<usize>();
    let mut expanded = Vec::with_capacity(
        prompt_token_ids.len() + total_replacement_len.saturating_sub(replacements.len()),
    );
    let mut offsets = Vec::with_capacity(replacements.len());
    let mut cursor = 0;

    for replacement in replacements {
        let offset = find_next_token(prompt_token_ids, PLACEHOLDER_TOKEN_ID, cursor).unwrap();
        let is_embed = replacement
            .tokens
            .iter()
            .map(|token| *token == EMBED_TOKEN_ID)
            .collect::<Vec<_>>();
        black_box(is_embed);

        expanded.extend_from_slice(&prompt_token_ids[cursor..offset]);
        offsets.push(expanded.len());
        expanded.extend_from_slice(&replacement.tokens);
        cursor = offset + 1;
    }

    expanded.extend_from_slice(&prompt_token_ids[cursor..]);
    *prompt_token_ids = expanded;

    offsets
}

fn bench_multimodal_prompt_expansion(c: &mut Criterion) {
    let cases = [
        make_case("4k_prompt_4_images_256_repl", 4 * 1024, 4, 256),
        make_case("32k_prompt_16_images_512_repl", 32 * 1024, 16, 512),
        make_case("128k_prompt_64_images_512_repl", 128 * 1024, 64, 512),
    ];

    let mut group = c.benchmark_group("multimodal_prompt_expansion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for case in cases {
        group.throughput(Throughput::Elements(case.final_token_count as u64));
        let mut splice_prompt = case.prompt.clone();
        let splice_offsets = expand_with_splice(&mut splice_prompt, &case.replacements);
        let mut single_pass_prompt = case.prompt.clone();
        let single_pass_offsets = expand_single_pass(&mut single_pass_prompt, &case.replacements);
        assert_eq!(single_pass_prompt, splice_prompt);
        assert_eq!(single_pass_offsets, splice_offsets);

        group.bench_function(format!("{}/splice_baseline", case.name), |b| {
            b.iter_batched(
                || case.prompt.clone(),
                |mut prompt| {
                    let offsets = expand_with_splice(&mut prompt, &case.replacements);
                    black_box((prompt, offsets));
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_function(format!("{}/single_pass", case.name), |b| {
            b.iter_batched(
                || case.prompt.clone(),
                |mut prompt| {
                    let offsets = expand_single_pass(&mut prompt, &case.replacements);
                    black_box((prompt, offsets));
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_multimodal_prompt_expansion);
criterion_main!(benches);
