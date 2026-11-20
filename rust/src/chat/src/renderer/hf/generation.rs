// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Support Hugging Face's `{% generation %}` blocks in MiniJinja chat templates.
//!
//! Transformers uses these blocks to track assistant-token masks via scoped call
//! blocks. Inference only needs their rendered text. MiniJinja lacks this tag, so
//! we rewrite `{% generation %}...{% endgeneration %}` to
//! `{% call __hf_generation() %}...{% endcall %}` before compiling the template.
//! The registered `__hf_generation` callback renders and returns `caller()`,
//! preserving the block's local scope just as Transformers' AssistantTracker does.
//!
//! We use MiniJinja's lexer to identify statement tags and replace only their
//! keyword spans. This preserves delimiters and whitespace controls (`-` / `+`),
//! while keeping literals, comments, and raw blocks intact. Original `generation`
//! and `call` pairings are checked before rewriting both to call blocks.

use minijinja::machinery::{Token, tokenize};
use minijinja::value::{Kwargs, Value};
use minijinja::{Error, ErrorKind, State};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallBlockKind {
    Call,
    Generation,
}

/// Rewrite Hugging Face's assistant-mask blocks as scoped MiniJinja call blocks.
pub(super) fn rewrite_generation_blocks(source: String) -> Result<String, Error> {
    let tokens = tokenize(&source, false, Default::default(), Default::default())
        .collect::<Result<Vec<_>, _>>()?;
    // Track native calls too: rewriting makes both block kinds use `endcall`,
    // so validate the original pairing before that distinction is lost.
    let mut blocks = Vec::new();
    let mut replacements = Vec::new();

    for (index, pair) in tokens.windows(2).enumerate() {
        let [(Token::BlockStart, _), (Token::Ident(name), span)] = pair else {
            continue;
        };
        match *name {
            "call" => blocks.push(CallBlockKind::Call),
            "endcall" if blocks.pop() == Some(CallBlockKind::Generation) => {
                return Err(Error::new(
                    ErrorKind::SyntaxError,
                    "expected endgeneration, got endcall",
                ));
            }
            "generation" | "endgeneration"
                if matches!(tokens.get(index + 2), Some((Token::BlockEnd, _))) =>
            {
                let replacement = if *name == "generation" {
                    blocks.push(CallBlockKind::Generation);
                    "call __hf_generation()"
                } else {
                    if blocks.pop() != Some(CallBlockKind::Generation) {
                        return Err(Error::new(
                            ErrorKind::SyntaxError,
                            "unexpected endgeneration",
                        ));
                    }
                    "endcall"
                };
                replacements.push((
                    span.start_offset as usize..span.end_offset as usize,
                    replacement,
                ));
            }
            _ => {}
        }
    }
    if blocks.contains(&CallBlockKind::Generation) {
        return Err(Error::new(
            ErrorKind::SyntaxError,
            "unclosed generation block",
        ));
    }

    // Replace only keyword spans, preserving delimiters and their whitespace controls.
    let mut source = source;
    for (span, replacement) in replacements.into_iter().rev() {
        source.replace_range(span, replacement);
    }
    Ok(source)
}

/// Render the caller with the same local scope as Transformers' AssistantTracker.
pub(super) fn render_generation(state: &State, kwargs: Kwargs) -> Result<Value, Error> {
    let caller: Value = kwargs.get("caller")?;
    kwargs.assert_all_used()?;
    caller.call(state, &[])
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use minijinja::Environment;

    use super::*;

    #[test]
    fn generation_blocks_reject_invalid_syntax_and_mismatched_end_tags() {
        let env = Environment::new();
        for template in [
            "{% generation %}unclosed",
            "{% endgeneration %}",
            "{% generation extra %}{% endgeneration %}",
            "{% generation %}{% endgeneration extra %}",
            "{% generation %}{% endcall %}",
            "{% call f() %}{% endgeneration %}",
            "{% generation %}{% endcall %}{% call f() %}{% endgeneration %}",
            "{% generation %}{% if true %}{% endgeneration %}{% endif %}",
        ] {
            let result = rewrite_generation_blocks(template.to_string())
                .and_then(|source| env.template_from_str(&source).map(|_| ()));
            assert!(result.is_err(), "accepted invalid template: {template}");
        }
    }

    #[test]
    fn rewrite_generation_blocks_preserves_hf_rendering_semantics() {
        let cases = [
            (
                "whitespace",
                "前  {%- generation -%}  中  {%- endgeneration -%}  后",
            ),
            (
                "keep_newlines",
                "A{%+ generation +%}\nB{%+ endgeneration +%}\nC",
            ),
            ("literal", "{{ '{% generation %}' }}"),
            ("raw", "{% raw %}{%- generation -%}{% endraw %}"),
            ("comment", "A{# {% generation %} #}B"),
            (
                "scope",
                "{% set x = 'outer' %}{% generation %}{% set x = 'inner' %}{{ x }}{% endgeneration %}{{ x }}",
            ),
            (
                "nested_calls",
                "{% macro wrap() %}[{{ caller() }}]{% endmacro %}{% generation %}A{% call wrap() %}{% generation %}B{% endgeneration %}{% endcall %}C{% endgeneration %}",
            ),
        ];
        let results = cases.map(|(name, source)| {
            let template = rewrite_generation_blocks(source.to_string()).unwrap();
            let mut env = Environment::new();
            env.set_trim_blocks(true);
            env.set_lstrip_blocks(true);
            env.add_function("__hf_generation", render_generation);
            let rendered =
                env.template_from_str(&template).unwrap().render(Value::UNDEFINED).unwrap();
            (name, rendered)
        });
        expect![[r#"
            [
                (
                    "whitespace",
                    "前中后",
                ),
                (
                    "keep_newlines",
                    "A\nB\nC",
                ),
                (
                    "literal",
                    "{% generation %}",
                ),
                (
                    "raw",
                    "{%- generation -%}",
                ),
                (
                    "comment",
                    "AB",
                ),
                (
                    "scope",
                    "innerouter",
                ),
                (
                    "nested_calls",
                    "A[B]C",
                ),
            ]
        "#]]
        .assert_debug_eq(&results);
    }
}
