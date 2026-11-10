// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Port of the helpers in `vllm/v1/structured_output/utils.py` that the
//! xgrammar backend and the manager use.

/// Drop speculative-decoding padding from a token block.
///
/// ngram and other speculative backends pad rejected draft positions with a
/// -1 sentinel. Structured-output grammars treat every entry as a real token
/// id, so the sentinels (and everything after the first one) are removed here,
/// before tokens reach any backend, rather than inside a single backend.
pub fn strip_speculative_padding(token_ids: &[i32]) -> &[i32] {
    match token_ids.iter().position(|&t| t < 0) {
        Some(i) => &token_ids[..i],
        None => token_ids,
    }
}

/// Check if grammar appears to use Lark syntax.
///
/// `grammar_is_likely_lark("rule: 'abc'")` is true and
/// `grammar_is_likely_lark("rule ::= 'abc'")` is false.
pub fn grammar_is_likely_lark(grammar_str: &str) -> bool {
    if grammar_str.is_empty() {
        return false;
    }
    grammar_str.split('\n').all(|line| {
        // Remove both comment styles (`re.sub(r"(#|//).*$", "", line)`).
        let code = match (line.find('#'), line.find("//")) {
            (Some(a), Some(b)) => &line[..a.min(b)],
            (Some(a), None) | (None, Some(a)) => &line[..a],
            (None, None) => line,
        };
        // Look for EBNF rule definition
        !code.contains("::=")
    })
}

/// `choice_as_grammar`: an EBNF grammar matching exactly one of `choice`.
pub fn choice_as_grammar<S: AsRef<str>>(choice: &[S]) -> String {
    /// Escape EBNF literals, including raw LF, CR, and NUL terminators.
    fn escape_ebnf_string(s: &str, out: &mut String) {
        for ch in s.chars() {
            match ch {
                '\\' => out.push_str(r"\\"),
                '"' => out.push_str(r#"\""#),
                '\n' => out.push_str(r"\n"),
                '\r' => out.push_str(r"\r"),
                '\t' => out.push_str(r"\t"),
                // Escape remaining C0 controls (U+0000-U+001F) and DEL (U+007F).
                c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                    out.push_str(&format!("\\u{:04x}", c as u32));
                }
                c => out.push(c),
            }
        }
    }

    let mut grammar = String::from("root ::= ");
    for (i, c) in choice.iter().enumerate() {
        if i > 0 {
            grammar.push_str(" | ");
        }
        grammar.push('"');
        escape_ebnf_string(c.as_ref(), &mut grammar);
        grammar.push('"');
    }
    grammar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lark_detection() {
        assert!(grammar_is_likely_lark("rule: 'abc'"));
        assert!(!grammar_is_likely_lark("rule ::= 'abc'"));
        assert!(!grammar_is_likely_lark(""));
        // A `::=` inside a comment does not count.
        assert!(grammar_is_likely_lark(
            "start: \"a\" # x ::= y\n// root ::= z"
        ));
        assert!(!grammar_is_likely_lark("# comment\n\nroot ::= \"a\""));
    }

    #[test]
    fn choice_grammar_escapes() {
        expect_test::expect![[r#"root ::= "a\"b" | "c\\d" | "\n\t\u0001\u007f" | "é""#]].assert_eq(
            &choice_as_grammar(&["a\"b", "c\\d", "\n\t\u{1}\u{7f}", "é"]),
        );
    }

    #[test]
    fn padding_is_stripped_from_first_sentinel() {
        assert_eq!(strip_speculative_padding(&[1, 2, -1, 3]), &[1, 2]);
        assert_eq!(strip_speculative_padding(&[1, 2]), &[1, 2]);
    }
}
