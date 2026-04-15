use std::collections::{HashSet, VecDeque};

use minijinja::machinery::ast::{Expr, ForLoop, Set, Stmt};
use minijinja::machinery::{WhitespaceConfig, parse};
use minijinja::syntax::SyntaxConfig;

/// Chat template content format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAi,
}

fn is_var_access(expr: &Expr, varname: &str) -> bool {
    matches!(expr, Expr::Var(v) if v.id == varname)
}

fn is_const_str(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
}

fn is_attr_access(expr: &Expr, varname: &str, key: &str) -> bool {
    match expr {
        Expr::GetItem(g) => is_var_access(&g.expr, varname) && is_const_str(&g.subscript_expr, key),
        Expr::GetAttr(g) => is_var_access(&g.expr, varname) && g.name == key,
        _ => false,
    }
}

fn is_var_or_elems_access(expr: &Expr, varname: &str, key: Option<&str>) -> bool {
    match expr {
        Expr::Filter(f) => f
            .expr
            .as_ref()
            .is_some_and(|inner| is_var_or_elems_access(inner, varname, key)),
        Expr::Test(t) => is_var_or_elems_access(&t.expr, varname, key),
        Expr::Slice(s) => is_var_or_elems_access(&s.expr, varname, key),
        _ => key.map_or_else(
            || is_var_access(expr, varname),
            |key| is_attr_access(expr, varname, key),
        ),
    }
}

fn visit_stmt<'a>(
    stmt: &'a Stmt<'a>,
    assignments: &mut Vec<&'a Set<'a>>,
    loops: &mut Vec<&'a ForLoop<'a>>,
) {
    match stmt {
        Stmt::Template(t) => {
            for child in &t.children {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::ForLoop(fl) => {
            loops.push(fl);
            for child in &fl.body {
                visit_stmt(child, assignments, loops);
            }
            for child in &fl.else_body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::IfCond(ic) => {
            for child in &ic.true_body {
                visit_stmt(child, assignments, loops);
            }
            for child in &ic.false_body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::WithBlock(wb) => {
            for child in &wb.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::Set(set_stmt) => assignments.push(set_stmt),
        Stmt::SetBlock(sb) => {
            for child in &sb.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::AutoEscape(ae) => {
            for child in &ae.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::FilterBlock(fb) => {
            for child in &fb.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::Block(b) => {
            for child in &b.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::Macro(m) => {
            for child in &m.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::CallBlock(cb) => {
            for child in &cb.macro_decl.body {
                visit_stmt(child, assignments, loops);
            }
        }
        _ => {}
    }
}

fn collect_assignments_and_loops<'a>(
    root: &'a Stmt<'a>,
) -> (Vec<&'a Set<'a>>, Vec<&'a ForLoop<'a>>) {
    let mut assignments = Vec::new();
    let mut loops = Vec::new();
    visit_stmt(root, &mut assignments, &mut loops);
    (assignments, loops)
}

fn iter_nodes_assign_var_or_elems(root: &Stmt<'_>, varname: &str) -> Vec<String> {
    let (assignments, _) = collect_assignments_and_loops(root);

    let mut discovered = vec![varname.to_string()];
    let mut seen = HashSet::from([varname.to_string()]);
    let mut related = VecDeque::from([varname.to_string()]);

    while let Some(related_varname) = related.pop_front() {
        for assign in &assignments {
            let Expr::Var(lhs) = &assign.target else {
                continue;
            };

            if is_var_or_elems_access(&assign.expr, &related_varname, None) {
                let lhs_name = lhs.id.to_string();
                if seen.insert(lhs_name.clone()) {
                    discovered.push(lhs_name.clone());
                    if lhs_name != related_varname {
                        related.push_back(lhs_name);
                    }
                }
            }
        }
    }

    discovered
}

fn iter_nodes_assign_messages_item(root: &Stmt<'_>) -> Vec<String> {
    let message_varnames = iter_nodes_assign_var_or_elems(root, "messages");
    let (_, loops) = collect_assignments_and_loops(root);

    let mut discovered = Vec::new();
    let mut seen = HashSet::new();

    for loop_ast in loops {
        let Expr::Var(target) = &loop_ast.target else {
            continue;
        };

        if message_varnames
            .iter()
            .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, None))
        {
            let target_name = target.id.to_string();
            if seen.insert(target_name.clone()) {
                discovered.push(target_name);
            }
        }
    }

    discovered
}

fn has_content_item_loop(root: &Stmt<'_>) -> bool {
    let message_varnames = iter_nodes_assign_messages_item(root);
    let (_, loops) = collect_assignments_and_loops(root);

    loops.into_iter().any(|loop_ast| {
        matches!(loop_ast.target, Expr::Var(_))
            && message_varnames
                .iter()
                .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, Some("content")))
    })
}

/// Detect the content format expected by a Jinja2 chat template based on AST analysis.
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return ChatTemplateContentFormat::String,
    };

    if has_content_item_loop(&ast) {
        ChatTemplateContentFormat::OpenAi
    } else {
        ChatTemplateContentFormat::String
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use expect_test::expect;

    use super::{ChatTemplateContentFormat, detect_chat_template_content_format};

    fn detect(template: &str) -> ChatTemplateContentFormat {
        detect_chat_template_content_format(template)
    }

    fn vllm_examples_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/templates/vllm_examples")
            .canonicalize()
            .expect("vLLM example template directory should exist locally")
    }

    fn read_vllm_example(relative_path: &str) -> String {
        fs::read_to_string(vllm_examples_dir().join(relative_path))
            .unwrap_or_else(|_| panic!("failed to read vLLM example template: {relative_path}"))
    }

    fn iter_vllm_example_template_paths() -> impl Iterator<Item = PathBuf> {
        let mut paths = fs::read_dir(vllm_examples_dir())
            .expect("failed to read vLLM example template directory")
            .map(|entry| {
                entry
                    .expect("failed to read vLLM example template dir entry")
                    .path()
            })
            .filter(|path| path.extension().is_some_and(|ext| ext == "jinja"))
            .collect::<Vec<_>>();
        paths.sort();
        paths.into_iter()
    }

    #[test]
    fn detects_string_template_without_content_loop() {
        assert_eq!(
            detect("{% for message in messages %}{{ message.content }}{% endfor %}"),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn detects_openai_template_with_direct_content_loop() {
        assert_eq!(
            detect(
                "{% for message in messages %}{% for content in message['content'] %}{{ content }}{% endfor %}{% endfor %}"
            ),
            ChatTemplateContentFormat::OpenAi
        );
    }

    #[test]
    fn detects_openai_template_with_messages_alias() {
        assert_eq!(
            detect(
                "{% set msgs = messages %}{% for message in msgs %}{% for content in message.content %}{{ content }}{% endfor %}{% endfor %}"
            ),
            ChatTemplateContentFormat::OpenAi
        );
    }

    #[test]
    fn does_not_detect_content_alias_loop_as_openai() {
        assert_eq!(
            detect(
                "{% for message in messages %}{% set parts = message.content %}{% for item in parts %}{{ item }}{% endfor %}{% endfor %}"
            ),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn does_not_treat_length_or_index_access_as_openai() {
        assert_eq!(
            detect("{% for message in messages %}{{ message.content|length }}{% endfor %}"),
            ChatTemplateContentFormat::String
        );
        assert_eq!(
            detect("{% for message in messages %}{{ message.content[0] }}{% endfor %}"),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn matches_vllm_example_template_formats() {
        let snapshot = iter_vllm_example_template_paths()
            .map(|path| {
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .expect("template file name should be valid UTF-8");
                let template = read_vllm_example(file_name);
                let format = detect(&template);
                format!("{file_name:50} => {format:?}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        expect![[r#"
            template_alpaca.jinja                              => String
            template_baichuan.jinja                            => String
            template_chatglm.jinja                             => String
            template_chatglm2.jinja                            => String
            template_chatml.jinja                              => String
            template_falcon.jinja                              => String
            template_falcon_180b.jinja                         => String
            template_inkbot.jinja                              => String
            template_teleflm.jinja                             => String
            tool_chat_template_deepseekr1.jinja                => String
            tool_chat_template_deepseekv3.jinja                => String
            tool_chat_template_deepseekv31.jinja               => String
            tool_chat_template_functiongemma.jinja             => String
            tool_chat_template_gemma3_pythonic.jinja           => OpenAi
            tool_chat_template_gemma4.jinja                    => OpenAi
            tool_chat_template_glm4.jinja                      => String
            tool_chat_template_granite.jinja                   => String
            tool_chat_template_granite_20b_fc.jinja            => String
            tool_chat_template_hermes.jinja                    => String
            tool_chat_template_hunyuan_a13b.jinja              => String
            tool_chat_template_internlm2_tool.jinja            => String
            tool_chat_template_llama3.1_json.jinja             => OpenAi
            tool_chat_template_llama3.2_json.jinja             => OpenAi
            tool_chat_template_llama3.2_pythonic.jinja         => String
            tool_chat_template_llama4_json.jinja               => OpenAi
            tool_chat_template_llama4_pythonic.jinja           => OpenAi
            tool_chat_template_minimax_m1.jinja                => OpenAi
            tool_chat_template_mistral.jinja                   => String
            tool_chat_template_mistral3.jinja                  => OpenAi
            tool_chat_template_mistral_parallel.jinja          => String
            tool_chat_template_phi4_mini.jinja                 => String
            tool_chat_template_qwen3coder.jinja                => String
            tool_chat_template_toolace.jinja                   => String
            tool_chat_template_xlam_llama.jinja                => String
            tool_chat_template_xlam_qwen.jinja                 => String"#]]
        .assert_eq(&snapshot);
    }
}
