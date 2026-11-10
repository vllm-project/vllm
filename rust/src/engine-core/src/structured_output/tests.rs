// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use expect_test::expect;
use vllm_xgrammar::{TokenizerInfo, VocabType};

use super::reasoner::ThinkTokenReasoner;
use super::*;

const VOCAB: [&str; 8] = ["{", "}", "1", "2", "a", "</s>", "<think>", "</think>"];
const EOS: i32 = 5;
const THINK_START: i32 = 6;
const THINK_END: i32 = 7;

fn manager(
    num_speculative_tokens: usize,
    async_grammar_compilation: bool,
) -> StructuredOutputManager {
    let info = TokenizerInfo::new(&VOCAB, VocabType::Raw, None, Some(&[EOS]), false).unwrap();
    let backend = XgrammarBackend::new(
        XgrammarBackendConfig {
            num_speculative_tokens,
            ..Default::default()
        },
        VOCAB.len(),
        &info,
        None,
    )
    .unwrap();
    StructuredOutputManager::new(
        StructuredOutputConfig {
            max_num_seqs: 2,
            num_speculative_tokens,
            is_diffusion: false,
            enable_in_reasoning: false,
            async_grammar_compilation,
            compile_threads: 1,
            fill_bitmask_threads: 1,
        },
        backend,
    )
}

fn view<'a>(id: &'a str, prompt: &'a [i32], all: &'a [i32]) -> RequestView<'a> {
    RequestView {
        request_id: id,
        use_structured_output: true,
        prompt_token_ids: Some(prompt),
        all_token_ids: all,
    }
}

/// Each bitmask row as the list of allowed token ids, `*` if all are.
fn allowed_rows(bitmask: &GrammarBitmask<'_>) -> Vec<String> {
    bitmask
        .data
        .chunks(bitmask.row_words)
        .map(|row| {
            if row.iter().all(|&w| w == -1) {
                return "*".to_owned();
            }
            let ids: Vec<_> = (0..VOCAB.len())
                .filter(|&t| row[t / 32] as u32 & (1 << (t % 32)) != 0)
                .map(|t| VOCAB[t])
                .collect();
            ids.join(" ")
        })
        .collect()
}

#[test]
fn bitmask_rows_follow_drafts_and_roll_back() {
    let mut m = manager(2, false);
    m.grammar_init(
        "r".into(),
        (StructuredOutputOptions::Regex, "[12]+".into()),
        None,
        None,
    );
    let prompt = [0];
    let req = view("r", &prompt, &prompt);
    // "1" is valid, "a" is not: the row after it allows everything.
    let drafts = [2, 4];
    let bitmask = m.grammar_bitmask(&[req], |_| Some(&drafts[..])).unwrap();
    expect![[r#"
        [
            "1 2",
            "1 2 </s>",
            "*",
        ]
    "#]]
    .assert_debug_eq(&allowed_rows(&bitmask));
    assert_eq!(m.grammar_mut("r").unwrap().num_processed_tokens(), 0);

    // `trim_spec_tokens` keeps the validated prefix plus one token.
    assert_eq!(m.trim_spec_tokens(req, vec![2, 4, 3]), vec![2, 4]);
    assert_eq!(m.trim_spec_tokens(req, vec![2, -1, -1]), vec![2]);

    assert!(m.accept_tokens(req, &[2, 3]));
    assert!(!m.accept_tokens(req, &[4]));
    assert!(m.accept_tokens(req, &[EOS]));
    assert!(m.grammar_mut("r").unwrap().is_terminated());
    // A terminated grammar leaves its rows unconstrained.
    let bitmask = m.grammar_bitmask(&[req], |_| None).unwrap();
    assert_eq!(allowed_rows(&bitmask)[0], "*");
}

#[test]
fn reasoning_gates_the_grammar() {
    let mut m = manager(0, false);
    let think = ThinkTokenReasoner {
        start_token_id: THINK_START,
        end_token_id: THINK_END,
    };
    m.grammar_init(
        "r".into(),
        (StructuredOutputOptions::Regex, "[12]+".into()),
        Some(vec![EOS]),
        Some(Box::new(think)),
    );
    let prompt = [0, THINK_START];
    let mut all = prompt.to_vec();

    let bitmask = m.grammar_bitmask(&[view("r", &prompt, &all)], |_| None).unwrap();
    assert_eq!(allowed_rows(&bitmask), ["*"]);
    // Tokens before the reasoning end bypass the grammar.
    all.push(4);
    assert!(m.accept_tokens(view("r", &prompt, &all), &[4]));
    // The grammar starts right after `</think>`.
    all.extend([THINK_END, 2]);
    assert!(m.accept_tokens(view("r", &prompt, &all), &[THINK_END, 2]));
    let bitmask = m.grammar_bitmask(&[view("r", &prompt, &all)], |_| None).unwrap();
    assert_eq!(allowed_rows(&bitmask), ["1 2 </s>"]);
}

#[test]
fn async_compilation_reports_ready_and_failed() {
    let mut m = manager(0, true);
    m.grammar_init(
        "ok".into(),
        (StructuredOutputOptions::JsonObject, String::new()),
        None,
        None,
    );
    m.grammar_init(
        "bad".into(),
        (StructuredOutputOptions::Regex, "(".into()),
        None,
        None,
    );
    let wait = |m: &mut StructuredOutputManager, id: &str| loop {
        match m.grammar_status(id) {
            GrammarStatus::Pending => std::thread::yield_now(),
            status => return status,
        }
    };
    assert_eq!(wait(&mut m, "ok"), GrammarStatus::Ready);
    assert!(matches!(wait(&mut m, "bad"), GrammarStatus::Failed { .. }));
    assert_eq!(m.grammar_status("unknown"), GrammarStatus::Pending);
}

#[test]
fn legacy_structural_tag_matches_pydantic_dump() {
    let s_tag = serde_json::json!({
        "structures": [{"begin": "<f=a>", "schema": {"type": "object", "properties": {"z": {"type": "integer"}, "a": {"type": "string"}}}, "end": "</f>"}],
        "triggers": ["<f="],
    });
    // `_get_structural_tag_str_from_args([tags, triggers], {})` in xgrammar 0.2.8.
    expect![[r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<f="],"tags":[{"type":"tag","begin":"<f=a>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"z":{"type":"integer"},"a":{"type":"string"}}},"style":"json","any_order":false,"max_whitespace_cnt":null,"excludes":[]},"end":"</f>"}],"at_least_one":false,"stop_after_first":false,"excludes":[]}}"#]]
        .assert_eq(&backend_xgrammar::legacy_structural_tag_json(&s_tag).unwrap());
}
