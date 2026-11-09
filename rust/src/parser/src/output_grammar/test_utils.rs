// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Readable renderings of structural-tag formats for test snapshots.

use std::fmt::Write;

use serde_json::{Map, Value};
use xgrammar_structural_tag::format::{EndBoundary, Format, TagBoundary, TagFormat, TokenValue};

/// Render `format` as an indented outline, one node per line.
///
/// Each node starts with its format type, and composite nodes list their
/// children one level deeper. Literal text is quoted in backticks with `\`,
/// `` ` ``, and control characters escaped, so markers read as written. Other
/// leaves are compact: `/regex/`, `text`, `json(SCHEMA)` or `STYLE(SCHEMA)`
/// for styled schemas, and `token ID` or ``token `TEXT` ``. Lists such as
/// exclusions and triggers are bracketed, and schemas read as types, see
/// [`schema`]. A single leaf child of `optional`, `star`, `plus`, `repeat`,
/// and dispatch rules stays on its parent's line, as does a tag's leaf
/// content between its boundaries. Options that keep their defaults are
/// omitted.
pub fn outline(format: &Format) -> String {
    let mut out = String::new();
    write(&mut out, 0, format_node(format));
    out
}

/// A child line of an outline node.
enum Child<'a> {
    Format(&'a Format),
    Tag(&'a TagFormat),
    Rule(String, &'a Format),
    Text(&'a str),
}

/// An outline node: its header line and children.
struct Node<'a> {
    header: String,
    children: Vec<Child<'a>>,
    /// Whether a single leaf child joins the header line.
    inline_leaf: bool,
}

impl<'a> Node<'a> {
    fn leaf(header: String) -> Self {
        Self {
            header,
            children: vec![],
            inline_leaf: false,
        }
    }

    fn branch(header: impl Into<String>, children: Vec<Child<'a>>) -> Self {
        Self {
            header: header.into(),
            children,
            inline_leaf: false,
        }
    }

    fn wrapper(header: impl Into<String>, child: Child<'a>) -> Self {
        Self {
            header: header.into(),
            children: vec![child],
            inline_leaf: true,
        }
    }
}

fn write(out: &mut String, depth: usize, node: Node<'_>) {
    let Node {
        mut header,
        children,
        inline_leaf,
    } = node;
    let mut children = children.into_iter().map(self::node).collect::<Vec<_>>();
    if inline_leaf
        && let [child] = &children[..]
        && child.children.is_empty()
    {
        let _ = write!(header, " {}", child.header);
        children.clear();
    }
    let _ = writeln!(out, "{}{header}", "  ".repeat(depth));
    for child in children {
        write(out, depth + 1, child);
    }
}

fn node(child: Child<'_>) -> Node<'_> {
    match child {
        Child::Format(format) => format_node(format),
        Child::Tag(tag) => tag_node(tag),
        Child::Rule(pattern, format) => {
            Node::wrapper(format!("on {pattern}"), Child::Format(format))
        }
        Child::Text(text) => Node::leaf(text.to_string()),
    }
}

fn format_node(format: &Format) -> Node<'_> {
    match format {
        Format::ConstString(format) => Node::leaf(quote(&format.value)),
        Format::Regex(format) => Node::leaf(format!("/{}/", format.pattern)),
        Format::AnyText(format) => {
            let mut header = "text".to_string();
            push_excluding(&mut header, format.excludes.iter().map(|text| quote(text)));
            push_limit(&mut header, "max_tokens", format.max_tokens);
            push_limit(&mut header, "max_chars", format.max_chars);
            Node::leaf(header)
        }
        Format::JsonSchema(format) => {
            let style = serde_json::to_value(format.style)
                .ok()
                .and_then(|style| style.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{:?}", format.style));
            let mut header = format!("{style}({})", schema(&format.json_schema));
            if format.any_order {
                header.push_str(" any_order");
            }
            if let Some(max) = format.max_whitespace_cnt {
                let _ = write!(header, " max_whitespace={max}");
            }
            push_excluding(&mut header, format.excludes.iter().map(|text| quote(text)));
            Node::leaf(header)
        }
        Format::Grammar(format) => {
            Node::branch("grammar", format.grammar.lines().map(Child::Text).collect())
        }
        Format::Token(format) => Node::leaf(format!("token {}", token(&format.token))),
        Format::ExcludeToken(format) => {
            let mut header = "token".to_string();
            push_excluding(&mut header, format.exclude_tokens.iter().map(token));
            Node::leaf(header)
        }
        Format::AnyTokens(format) => {
            let mut header = "tokens".to_string();
            push_excluding(&mut header, format.exclude_tokens.iter().map(token));
            push_limit(&mut header, "max_tokens", format.max_tokens);
            Node::leaf(header)
        }
        Format::Sequence(format) => Node::branch(
            "sequence",
            format.elements.iter().map(Child::Format).collect(),
        ),
        Format::Or(format) => {
            Node::branch("or", format.elements.iter().map(Child::Format).collect())
        }
        Format::Optional(format) => Node::wrapper("optional", Child::Format(&format.content)),
        Format::Plus(format) => Node::wrapper("plus", Child::Format(&format.content)),
        Format::Star(format) => Node::wrapper("star", Child::Format(&format.content)),
        Format::Repeat(format) => {
            let max = if format.max < 0 {
                String::new()
            } else {
                format.max.to_string()
            };
            Node::wrapper(
                format!("repeat {}..{max}", format.min),
                Child::Format(&format.content),
            )
        }
        Format::Tag(tag) => tag_node(tag),
        Format::TriggeredTags(format) => {
            let triggers = format.triggers.iter().map(|trigger| quote(trigger));
            let mut header = format!(
                "triggered_tags [{}]",
                triggers.collect::<Vec<_>>().join(", ")
            );
            push_excluding(&mut header, format.excludes.iter().map(|text| quote(text)));
            push_flags(&mut header, format.at_least_one, format.stop_after_first);
            Node::branch(header, format.tags.iter().map(Child::Tag).collect())
        }
        Format::TokenTriggeredTags(format) => {
            let triggers = format.trigger_tokens.iter().map(token);
            let mut header = format!(
                "token_triggered_tags [{}]",
                triggers.collect::<Vec<_>>().join(", ")
            );
            push_excluding(&mut header, format.exclude_tokens.iter().map(token));
            push_flags(&mut header, format.at_least_one, format.stop_after_first);
            Node::branch(header, format.tags.iter().map(Child::Tag).collect())
        }
        Format::TagsWithSeparator(format) => {
            let mut header = format!("tags_with_separator {}", quote(&format.separator));
            push_flags(&mut header, format.at_least_one, format.stop_after_first);
            Node::branch(header, format.tags.iter().map(Child::Tag).collect())
        }
        Format::Dispatch(format) => {
            let mut header = "dispatch".to_string();
            if !format.r#loop {
                header.push_str(" once");
            }
            push_excluding(&mut header, format.excludes.iter().map(|text| quote(text)));
            Node::branch(
                header,
                format
                    .rules
                    .iter()
                    .map(|(pattern, format)| Child::Rule(quote(pattern), format))
                    .collect(),
            )
        }
        Format::TokenDispatch(format) => {
            let mut header = "token_dispatch".to_string();
            if !format.r#loop {
                header.push_str(" once");
            }
            push_excluding(&mut header, format.exclude_tokens.iter().map(token));
            Node::branch(
                header,
                format
                    .rules
                    .iter()
                    .map(|(trigger, format)| Child::Rule(token(trigger), format))
                    .collect(),
            )
        }
    }
}

/// `tag BEGIN CONTENT END` for leaf content, or `tag BEGIN .. END` above the
/// content's own node.
fn tag_node(tag: &TagFormat) -> Node<'_> {
    let begin = match &tag.begin {
        TagBoundary::Text(text) => quote(text),
        TagBoundary::Token(boundary) => format!("token {}", token(&boundary.token)),
    };
    let end = match &tag.end {
        EndBoundary::Text(text) => quote(text),
        EndBoundary::Texts(texts) => {
            texts.iter().map(|text| quote(text)).collect::<Vec<_>>().join(" | ")
        }
        EndBoundary::Token(boundary) => format!("token {}", token(&boundary.token)),
    };
    let content = format_node(&tag.content);
    if content.children.is_empty() {
        Node::leaf(format!("tag {begin} {} {end}", content.header))
    } else {
        Node::branch(
            format!("tag {begin} .. {end}"),
            vec![Child::Format(&tag.content)],
        )
    }
}

/// Schema keywords that do not constrain the generated text.
const ANNOTATIONS: &[&str] = &[
    "$comment",
    "$id",
    "$schema",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
];

/// Render a JSON schema as a TypeScript-like type.
///
/// Objects are `{ key: T, optional?: T }`, with `...` for any additional
/// property and `...: T` for additional properties under `T`. Arrays are
/// `T[]`. Unions, type arrays, and enums join their options with `|`,
/// `oneOf` and `allOf` keep their names, local references show the
/// definition name, a `type` next to a reference intersects it as
/// `NAME & T`, and `true` / `false` are `any` / `never`. Every other
/// keyword follows the type as `(key=VALUE, ...)`, and definitions follow the
/// root as `where NAME = T, ...`. Annotations such as `description` are
/// dropped.
pub fn schema(schema: &Value) -> String {
    typed(schema).text
}

/// A rendered type, and whether it is a top-level union or intersection that
/// an operand position must parenthesize.
struct Typed {
    text: String,
    compound: bool,
}

impl Typed {
    fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            compound: false,
        }
    }

    fn compound(text: String) -> Self {
        Self {
            text,
            compound: true,
        }
    }

    fn union(options: Vec<String>) -> Self {
        Self {
            compound: options.len() > 1,
            text: options.join(" | "),
        }
    }

    /// The type as an operand of `&` or of a suffix such as `[]` or `(...)`.
    fn grouped(self) -> String {
        if self.compound {
            format!("({})", self.text)
        } else {
            self.text
        }
    }
}

fn typed(schema: &Value) -> Typed {
    let schema = match schema {
        Value::Bool(true) => return Typed::plain("any"),
        Value::Bool(false) => return Typed::plain("never"),
        Value::Object(schema) => schema,
        schema => return Typed::plain(schema.to_string()),
    };
    let mut rendered = SchemaType::new(schema).render();
    let definitions = ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| schema.get(key)?.as_object())
        .flatten()
        .map(|(name, definition)| format!("{name} = {}", self::schema(definition)))
        .collect::<Vec<_>>();
    if !definitions.is_empty() {
        rendered = Typed::plain(format!(
            "{} where {}",
            rendered.text,
            definitions.join(", ")
        ));
    }
    rendered
}

/// One schema object being rendered, with the keywords its type consumed.
struct SchemaType<'a> {
    schema: &'a Map<String, Value>,
    consumed: Vec<&'static str>,
}

impl<'a> SchemaType<'a> {
    fn new(schema: &'a Map<String, Value>) -> Self {
        Self {
            schema,
            consumed: vec!["$defs", "definitions"],
        }
    }

    fn take(&mut self, key: &'static str) -> Option<&'a Value> {
        let value = self.schema.get(key)?;
        self.consumed.push(key);
        Some(value)
    }

    fn render(mut self) -> Typed {
        let mut rendered = self.body();
        let rest = self
            .schema
            .iter()
            .filter(|(key, _)| {
                !self.consumed.contains(&key.as_str()) && !ANNOTATIONS.contains(&key.as_str())
            })
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>();
        if !rest.is_empty() {
            rendered = Typed::plain(format!("{}({})", rendered.grouped(), rest.join(", ")));
        }
        rendered
    }

    fn body(&mut self) -> Typed {
        if let Some(Value::String(reference)) = self.take("$ref") {
            let name = ["#/$defs/", "#/definitions/"]
                .into_iter()
                .find_map(|prefix| reference.strip_prefix(prefix))
                .unwrap_or(reference);
            // Keywords next to `$ref` apply together with it.
            return match self.take("type") {
                Some(type_name) => {
                    Typed::compound(format!("{name} & {}", type_names(type_name).grouped()))
                }
                None => Typed::plain(name),
            };
        }
        if let Some(value) = self.take("const") {
            self.take_redundant_type([value]);
            return Typed::plain(value.to_string());
        }
        if let Some(Value::Array(values)) = self.take("enum") {
            self.take_redundant_type(values);
            return Typed::union(values.iter().map(Value::to_string).collect());
        }
        if let Some(Value::Array(options)) = self.take("anyOf") {
            return Typed::union(options.iter().map(schema).collect());
        }
        for combinator in ["oneOf", "allOf"] {
            if let Some(Value::Array(options)) = self.take(combinator) {
                let options = options.iter().map(schema).collect::<Vec<_>>();
                return Typed::plain(format!("{combinator}({})", options.join(", ")));
            }
        }
        let is = |name: &str| self.schema.get("type").is_none_or(|type_name| type_name == name);
        if is("object")
            && (self.schema.contains_key("properties")
                || self.schema.contains_key("additionalProperties"))
        {
            self.take("type");
            return Typed::plain(self.object());
        }
        if is("array") && self.schema.contains_key("items") {
            self.take("type");
            let items = self.take("items").map_or_else(|| Typed::plain("any"), typed);
            return Typed::plain(format!("{}[]", items.grouped()));
        }
        match self.take("type") {
            Some(type_name) => type_names(type_name),
            None => Typed::plain("any"),
        }
    }

    /// Consume `type` when every literal already has that type.
    fn take_redundant_type<'v>(&mut self, values: impl IntoIterator<Item = &'v Value>) {
        let Some(Value::String(type_name)) = self.schema.get("type") else {
            return;
        };
        let matches = |value: &Value| match value {
            Value::String(_) => type_name == "string",
            Value::Number(number) => {
                type_name == "number" || (type_name == "integer" && !number.is_f64())
            }
            Value::Bool(_) => type_name == "boolean",
            Value::Null => type_name == "null",
            Value::Object(_) => type_name == "object",
            Value::Array(_) => type_name == "array",
        };
        if values.into_iter().all(matches) {
            self.take("type");
        }
    }

    fn object(&mut self) -> String {
        let required = self
            .take("required")
            .and_then(Value::as_array)
            .map(|required| required.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = self.take("properties").and_then(Value::as_object);
        let mut fields = properties
            .into_iter()
            .flatten()
            .map(|(key, value)| {
                let optional = if required.contains(&key.as_str()) {
                    ""
                } else {
                    "?"
                };
                format!("{key}{optional}: {}", schema(value))
            })
            .collect::<Vec<_>>();
        // Required keys without a declared schema.
        fields.extend(
            required
                .iter()
                .filter(|key| properties.is_none_or(|properties| !properties.contains_key(**key)))
                .map(|key| format!("{key}: any")),
        );
        match self.take("additionalProperties") {
            Some(Value::Bool(false)) | None => {}
            Some(Value::Bool(true)) => fields.push("...".to_string()),
            Some(additional) => fields.push(format!("...: {}", schema(additional))),
        }
        if fields.is_empty() {
            "{}".to_string()
        } else {
            format!("{{ {} }}", fields.join(", "))
        }
    }
}

/// A `type` keyword's type names.
fn type_names(type_name: &Value) -> Typed {
    match type_name {
        Value::String(name) if name == "array" => Typed::plain("any[]"),
        Value::String(name) => Typed::plain(name.clone()),
        Value::Array(names) => Typed::union(
            names
                .iter()
                .map(|name| name.as_str().map_or_else(|| name.to_string(), str::to_string))
                .collect(),
        ),
        type_name => Typed::plain(type_name.to_string()),
    }
}

fn quote(text: &str) -> String {
    let mut quoted = String::with_capacity(text.len() + 2);
    quoted.push('`');
    for char in text.chars() {
        match char {
            '\\' => quoted.push_str("\\\\"),
            '`' => quoted.push_str("\\`"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            char if char.is_control() => {
                let _ = write!(quoted, "\\u{{{:x}}}", char as u32);
            }
            char => quoted.push(char),
        }
    }
    quoted.push('`');
    quoted
}

fn token(token: &TokenValue) -> String {
    match token {
        TokenValue::Id(id) => id.to_string(),
        TokenValue::Text(text) => quote(text),
    }
}

fn push_excluding(header: &mut String, excludes: impl Iterator<Item = String>) {
    let excludes = excludes.collect::<Vec<_>>();
    if !excludes.is_empty() {
        let _ = write!(header, " excluding [{}]", excludes.join(", "));
    }
}

fn push_limit(header: &mut String, name: &str, limit: Option<u32>) {
    if let Some(limit) = limit {
        let _ = write!(header, " {name}={limit}");
    }
}

fn push_flags(header: &mut String, at_least_one: bool, stop_after_first: bool) {
    if at_least_one {
        header.push_str(" at_least_one");
    }
    if stop_after_first {
        header.push_str(" stop_after_first");
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;
    use xgrammar_structural_tag::format::{
        AnyTokensFormat, DispatchFormat, ExcludeTokenFormat, GrammarFormat, JsonSchemaFormat,
        JsonSchemaStyle, TokenBoundary, TokenDispatchFormat, TokenFormat, TokenTriggeredTagsFormat,
    };

    use super::*;

    #[test]
    fn outline_renders_every_format_type() {
        let format = Format::sequence(vec![
            Format::Token(TokenFormat {
                token: TokenValue::Id(7),
            }),
            Format::ExcludeToken(ExcludeTokenFormat {
                exclude_tokens: vec![TokenValue::Id(1), TokenValue::Text("<eos>".to_string())],
            }),
            Format::AnyTokens(AnyTokensFormat {
                exclude_tokens: vec![TokenValue::Id(2)],
                max_tokens: Some(16),
            }),
            Format::Grammar(GrammarFormat {
                grammar: "root ::= a\na ::= \"x\"".to_string(),
            }),
            Format::repeat(Format::regex("[0-9]"), 1, -1),
            Format::JsonSchema(
                JsonSchemaFormat::new(json!({ "type": "string" }))
                    .with_style(JsonSchemaStyle::KimiK3Xml)
                    .with_any_order(true)
                    .with_max_whitespace_cnt(Some(2))
                    .with_excludes(&["</arg>"]),
            ),
            Format::TokenTriggeredTags(TokenTriggeredTagsFormat {
                trigger_tokens: vec![TokenValue::Id(9)],
                tags: vec![TagFormat::new(
                    TokenBoundary::new(9),
                    Format::optional(Format::const_string("`\t")),
                    vec!["</a>", "</b>"],
                )],
                exclude_tokens: vec![],
                at_least_one: false,
                stop_after_first: true,
            }),
            Format::Dispatch(DispatchFormat {
                rules: vec![(
                    "<call>".to_string(),
                    Format::sequence(vec![Format::any_text(), Format::const_string("</call>")]),
                )],
                r#loop: false,
                excludes: vec!["<stop>".to_string()],
            }),
            Format::TokenDispatch(TokenDispatchFormat {
                rules: vec![(
                    TokenValue::Id(3),
                    Format::JsonSchema(JsonSchemaFormat::new(json!(true))),
                )],
                r#loop: true,
                exclude_tokens: vec![],
            }),
        ]);

        expect![[r#"
            sequence
              token 7
              token excluding [1, `<eos>`]
              tokens excluding [2] max_tokens=16
              grammar
                root ::= a
                a ::= "x"
              repeat 1.. /[0-9]/
              kimi_k3_xml(string) any_order max_whitespace=2 excluding [`</arg>`]
              token_triggered_tags [9] stop_after_first
                tag token 9 .. `</a>` | `</b>`
                  optional `\`\t`
              dispatch once excluding [`<stop>`]
                on `<call>`
                  sequence
                    text
                    `</call>`
              token_dispatch
                on 3 json(any)
        "#]]
        .assert_eq(&outline(&format));
    }

    #[test]
    fn schema_renders_common_shapes_as_types() {
        let rendered = schema(&json!({
            "$defs": { "place": { "type": "object", "properties": { "city": { "type": "string" } } } },
            "type": "object",
            "description": "ignored",
            "properties": {
                "location": { "type": "string" },
                "id": { "type": ["integer", "null"] },
                "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                "tags": { "type": "array", "items": { "anyOf": [{ "type": "string" }, { "type": "number" }] } },
                "place": { "$ref": "#/$defs/place" },
                "days": { "type": "integer", "minimum": 1 },
                "free": { "type": "object", "additionalProperties": true }
            },
            "required": ["location", "id"],
            "additionalProperties": false
        }));

        expect![[r#"{ location: string, id: integer | null, unit?: "celsius" | "fahrenheit", tags?: (string | number)[], place?: place, days?: integer(minimum=1), free?: { ... } } where place = { city?: string }"#]].assert_eq(&rendered);
    }

    #[test]
    fn schema_keeps_other_keywords_after_the_type() {
        let rendered = schema(&json!({
            "type": "object",
            "properties": {
                "code": { "type": "string", "pattern": "[A-Z]{3}" },
                "ids": { "type": "array", "items": { "type": ["integer", "null"] }, "maxItems": 3 },
                "place": { "$ref": "#/$defs/place", "type": "object" },
                "either": { "not": { "type": "null" } }
            },
            "required": ["code", "tenant"],
            "additionalProperties": { "type": "boolean" },
            "minProperties": 1
        }));

        expect![[r#"{ code: string(pattern="[A-Z]{3}"), ids?: (integer | null)[](maxItems=3), place?: place & object, either?: any(not={"type":"null"}), tenant: any, ...: boolean }(minProperties=1)"#]].assert_eq(&rendered);
    }
}
