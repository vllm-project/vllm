// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Random dataset specialized for scoring/rerank benchmarks: each request is
//! one query plus a batch of documents. Mirrors Python's
//! `RandomDatasetForReranking`.
//!
//! With `is_reranker` (default): the query and each document share the
//! request's token budget (`query + sep + doc ~= input_len`), and every
//! batched request counts the query once per document pair.
//! With `--no-reranker` (embedding-based scoring): the query is just another
//! embedding input occupying the first batch slot.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::SampleRequest;
use crate::config::RangeRatio;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

pub fn generate_random_rerank_dataset(
    tokenizer: &TokenizerKind,
    num_requests: usize,
    input_len: usize,
    range_ratio: RangeRatio,
    seed: u64,
    request_id_prefix: &str,
    batch_size: usize,
    is_reranker: bool,
) -> Result<Vec<SampleRequest>> {
    let allowed_tokens = tokenizer.get_allowed_tokens();
    if allowed_tokens.is_empty() {
        return Err(BenchError::Tokenizer("No allowed tokens found".into()));
    }
    let allowed_ref = &allowed_tokens;

    let num_special = tokenizer.num_special_tokens_to_add();
    let real_input_len = input_len.saturating_sub(num_special);

    let n_sep_tokens = usize::from(is_reranker);
    let query_len_param = if is_reranker {
        (real_input_len / 2).saturating_sub(n_sep_tokens)
    } else {
        real_input_len
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let sample = |rng: &mut StdRng, (low, high): (usize, usize)| -> usize {
        if low == high {
            low
        } else {
            rng.random_range(low..=high)
        }
    };

    // One query length for the whole run, like Python.
    let query_len = sample(&mut rng, range_ratio.input_bounds(query_len_param));

    // --no-reranker folds the query into the first batch slot.
    let (num_docs, docs_per_batch, doc_len_param) = if is_reranker {
        let doc_len = real_input_len.saturating_sub(query_len).saturating_sub(n_sep_tokens);
        (num_requests, batch_size, doc_len)
    } else {
        (num_requests - 1, batch_size - 1, real_input_len)
    };
    if doc_len_param == 0 {
        return Err(BenchError::Config(format!(
            "random-rerank: --random-input-len {input_len} leaves no budget for documents \
             (query_len={query_len})"
        )));
    }

    // Pre-sample per-document lengths and offsets deterministically.
    let doc_bounds = range_ratio.input_bounds(doc_len_param);
    let doc_params: Vec<(usize, usize)> = (0..num_docs)
        .map(|_| {
            (
                sample(&mut rng, doc_bounds),
                rng.random_range(0..allowed_ref.len()),
            )
        })
        .collect();
    let query_offset = rng.random_range(0..allowed_ref.len());

    // Exact-length text: token sequence -> decode -> re-encode -> truncate -> decode.
    let gen_text = |target: usize, offset: usize, index: usize| -> Result<(Arc<str>, usize)> {
        let at_len = allowed_ref.len();
        let tokens: Vec<u32> =
            (0..target).map(|j| allowed_ref[(offset + index + j) % at_len]).collect();
        let text = tokenizer.decode(&tokens, true)?;
        let mut re_encoded = tokenizer.encode(&text, false)?;
        re_encoded.truncate(target);
        let final_text = tokenizer.decode(&re_encoded, true)?;
        Ok((Arc::from(final_text), re_encoded.len()))
    };

    let (query_prompt, query_input_len) = gen_text(query_len, query_offset, 0)?;

    let docs: Vec<(Arc<str>, usize)> = doc_params
        .par_iter()
        .enumerate()
        .map(|(i, (len, offset))| gen_text(*len, *offset, i + 1))
        .collect::<Result<Vec<_>>>()?;

    // Batch documents; every request is [query, doc1, doc2, ...].
    let rid_prefix = request_id_prefix.to_string();
    let requests = docs
        .chunks(docs_per_batch)
        .enumerate()
        .map(|(batch_idx, batch)| {
            let query_contrib = if is_reranker {
                (query_input_len + n_sep_tokens) * batch.len()
            } else {
                query_input_len
            };
            let mut prompt_list: Vec<Arc<str>> = Vec::with_capacity(batch.len() + 1);
            prompt_list.push(query_prompt.clone());
            prompt_list.extend(batch.iter().map(|(text, _)| text.clone()));
            SampleRequest {
                prompt_list: Some(Arc::from(prompt_list)),
                prompt_len: query_contrib + batch.iter().map(|(_, len)| len).sum::<usize>(),
                expected_output_len: 0,
                request_id: Some(format!("{rid_prefix}{batch_idx}")),
                ..Default::default()
            }
        })
        .collect();

    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gpt2 via built-in tiktoken encoding — loads without network access.
    fn test_tokenizer() -> TokenizerKind {
        TokenizerKind::Tiktoken(
            crate::tiktoken::load_builtin_tiktoken("gpt2")
                .expect("gpt2 built-in tiktoken should always load without network"),
        )
    }

    fn fixed_ratio() -> RangeRatio {
        RangeRatio::parse("0.0").unwrap()
    }

    #[test]
    fn test_random_rerank_reranker_mode() {
        let tok = test_tokenizer();
        // 6 docs in batches of 3 -> 2 requests of [query, d1, d2, d3]
        let reqs = generate_random_rerank_dataset(&tok, 6, 128, fixed_ratio(), 0, "t-", 3, true)
            .expect("generation should succeed");
        assert_eq!(reqs.len(), 2);
        for r in &reqs {
            let list = r.prompt_list.as_ref().expect("prompt_list must be set");
            assert_eq!(list.len(), 4);
            assert_eq!(r.expected_output_len, 0);
            assert!(r.prompt_len > 0);
        }
        // Same query shared across requests
        assert_eq!(
            reqs[0].prompt_list.as_ref().unwrap()[0],
            reqs[1].prompt_list.as_ref().unwrap()[0]
        );
        // Reranker budget: query+sep+doc pairs stay near input_len per pair
        // (query ~63, doc ~64 for input_len=128, gpt2 has no special tokens)
        let list = reqs[0].prompt_list.as_ref().unwrap();
        let query_tokens = tok.encode(&list[0], false).unwrap().len();
        assert!(query_tokens <= 64, "query too long: {query_tokens}");
    }

    #[test]
    fn test_random_rerank_no_reranker_mode() {
        let tok = test_tokenizer();
        // no-reranker: query occupies first slot; 5 non-query docs in batches of 2
        let reqs = generate_random_rerank_dataset(&tok, 6, 64, fixed_ratio(), 0, "t-", 3, false)
            .expect("generation should succeed");
        // 6-1=5 docs, batches of 3-1=2 -> 3 requests
        assert_eq!(reqs.len(), 3);
        assert_eq!(reqs[0].prompt_list.as_ref().unwrap().len(), 3);
    }
}
