use minijinja::machinery::ast::{Expr, Stmt};
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

/// Flags tracking which OpenAI-style patterns we've seen.
#[derive(Default, Debug, Clone, Copy)]
struct Flags {
    saw_iteration: bool,
    saw_structure: bool,
    saw_assignment: bool,
    saw_macro: bool,
}

impl Flags {
    fn any(self) -> bool {
        // `saw_assignment` alone (e.g. `set content = message.content`) is not sufficient to
        // classify as OpenAI format. Many string-format templates use this pattern to extract
        // content into a local variable, then check `content is string`.
        self.saw_iteration || self.saw_structure || self.saw_macro
    }
}

/// Single-pass AST detector with scope tracking.
struct Detector<'a> {
    ast: &'a Stmt<'a>,
    /// Message loop vars currently in scope (e.g., `message`, `m`, `msg`).
    scope: std::collections::VecDeque<String>,
    scope_set: std::collections::HashSet<String>,
    flags: Flags,
}

impl<'a> Detector<'a> {
    fn new(ast: &'a Stmt<'a>) -> Self {
        Self {
            ast,
            scope: std::collections::VecDeque::new(),
            scope_set: std::collections::HashSet::new(),
            flags: Flags::default(),
        }
    }

    fn run(mut self) -> Flags {
        self.walk_stmt(self.ast);
        self.flags
    }

    fn push_scope(&mut self, var: String) {
        self.scope.push_back(var.clone());
        self.scope_set.insert(var);
    }

    fn pop_scope(&mut self) {
        if let Some(v) = self.scope.pop_back() {
            self.scope_set.remove(&v);
        }
    }

    fn is_var_access(expr: &Expr, varname: &str) -> bool {
        matches!(expr, Expr::Var(v) if v.id == varname)
    }

    fn is_const_str(expr: &Expr, value: &str) -> bool {
        matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
    }

    fn is_numeric_const(expr: &Expr) -> bool {
        matches!(expr, Expr::Const(c) if c.value.is_number())
    }

    /// Check if expr is varname.content or varname["content"].
    fn is_var_dot_content(expr: &Expr, varname: &str) -> bool {
        match expr {
            Expr::GetAttr(g) => Self::is_var_access(&g.expr, varname) && g.name == "content",
            Expr::GetItem(g) => {
                Self::is_var_access(&g.expr, varname)
                    && Self::is_const_str(&g.subscript_expr, "content")
            }
            Expr::Filter(f) => f
                .expr
                .as_ref()
                .is_some_and(|e| Self::is_var_dot_content(e, varname)),
            Expr::Test(t) => Self::is_var_dot_content(&t.expr, varname),
            _ => false,
        }
    }

    /// Check if expr accesses `.content` on any variable in scope, or any descendant of it.
    fn is_any_scope_var_content(&self, expr: &Expr) -> bool {
        let mut current_expr = expr;
        loop {
            if self
                .scope_set
                .iter()
                .any(|v| Self::is_var_dot_content(current_expr, v))
            {
                return true;
            }
            match current_expr {
                Expr::GetAttr(g) => current_expr = &g.expr,
                Expr::GetItem(g) => current_expr = &g.expr,
                _ => return false,
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Template(t) => {
                for ch in &t.children {
                    self.walk_stmt(ch);
                }
            }
            Stmt::ForLoop(fl) => {
                if let Expr::Var(iter) = &fl.iter
                    && iter.id == "messages"
                    && let Expr::Var(target) = &fl.target
                {
                    self.push_scope(target.id.to_string());
                }

                if self.is_any_scope_var_content(&fl.iter) {
                    self.flags.saw_iteration = true;
                }
                if matches!(&fl.iter, Expr::Var(v) if v.id == "content") {
                    self.flags.saw_iteration = true;
                }

                for b in &fl.body {
                    self.walk_stmt(b);
                }

                if let Expr::Var(iter) = &fl.iter
                    && iter.id == "messages"
                    && matches!(&fl.target, Expr::Var(_))
                {
                    self.pop_scope();
                }
            }
            Stmt::IfCond(ic) => {
                self.inspect_expr_for_structure(&ic.expr);

                for b in &ic.true_body {
                    self.walk_stmt(b);
                }
                for b in &ic.false_body {
                    self.walk_stmt(b);
                }
            }
            Stmt::EmitExpr(e) => {
                self.inspect_expr_for_structure(&e.expr);
            }
            Stmt::Set(s)
                if Self::is_var_access(&s.target, "content")
                    && self.is_any_scope_var_content(&s.expr) =>
            {
                self.flags.saw_assignment = true;
            }
            Stmt::Macro(m) => {
                let mut has_type_check = false;
                let mut has_loop = false;
                Self::scan_macro_body(&m.body, &mut has_type_check, &mut has_loop);
                if has_type_check && has_loop {
                    self.flags.saw_macro = true;
                }
            }
            _ => {}
        }
    }

    fn inspect_expr_for_structure(&mut self, expr: &Expr) {
        if self.flags.saw_structure {
            return;
        }

        match expr {
            Expr::GetItem(gi)
                if (matches!(&gi.expr, Expr::Var(v) if v.id == "content")
                    || self.is_any_scope_var_content(&gi.expr))
                    && Self::is_numeric_const(&gi.subscript_expr) =>
            {
                self.flags.saw_structure = true;
            }
            Expr::Filter(f) => {
                if f.name == "length" {
                    if let Some(inner) = &f.expr {
                        let inner_ref: &Expr = inner;
                        let is_content_var = matches!(inner_ref, Expr::Var(v) if v.id == "content");
                        if is_content_var || self.is_any_scope_var_content(inner_ref) {
                            self.flags.saw_structure = true;
                        }
                    }
                } else if let Some(inner) = &f.expr {
                    let inner_ref: &Expr = inner;
                    self.inspect_expr_for_structure(inner_ref);
                }
            }
            Expr::Test(t) => self.inspect_expr_for_structure(&t.expr),
            Expr::GetAttr(g) => {
                self.inspect_expr_for_structure(&g.expr);
            }
            Expr::BinOp(op) => {
                self.inspect_expr_for_structure(&op.left);
                self.inspect_expr_for_structure(&op.right);
            }
            Expr::UnaryOp(op) => {
                self.inspect_expr_for_structure(&op.expr);
            }
            _ => {}
        }
    }

    fn scan_macro_body(body: &[Stmt], has_type_check: &mut bool, has_loop: &mut bool) {
        for s in body {
            if *has_type_check && *has_loop {
                return;
            }

            match s {
                Stmt::IfCond(ic) => {
                    if matches!(&ic.expr, Expr::Test(_)) {
                        *has_type_check = true;
                    }
                    Self::scan_macro_body(&ic.true_body, has_type_check, has_loop);
                    Self::scan_macro_body(&ic.false_body, has_type_check, has_loop);
                }
                Stmt::ForLoop(fl) => {
                    *has_loop = true;
                    Self::scan_macro_body(&fl.body, has_type_check, has_loop);
                }
                Stmt::Template(t) => {
                    Self::scan_macro_body(&t.children, has_type_check, has_loop);
                }
                _ => {}
            }
        }
    }
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

    let flags = Detector::new(&ast).run();
    if flags.any() {
        ChatTemplateContentFormat::OpenAi
    } else {
        ChatTemplateContentFormat::String
    }
}
