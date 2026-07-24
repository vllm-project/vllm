# Rust Frontend

The vLLM Rust frontend is an experimental drop-in alternative to the Python API
server, living under `rust/` as a Cargo workspace. This guide covers local
development: setting up the toolchain, building, testing, and the recurring
flows for adding parsers and other contributions.

For the architecture and the high-level runtime model, start with
[`rust/README.md`](https://github.com/vllm-project/vllm/blob/main/rust/README.md).
For the up-to-date feature-parity tracker and design decisions, see the
[Rust Frontend Feature Parity roadmap](https://github.com/vllm-project/vllm/issues/44280)
and the [RFC](https://github.com/vllm-project/vllm/issues/40846).

## Local development setup

### Toolchain

The Rust toolchain is pinned through `rust-toolchain.toml` at the repo root.
`rustup` reads that file automatically when you run any `cargo` command inside
the workspace; you do not need to install the toolchain manually if `rustup`
is available.

If you do not have `rustup` yet:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --profile minimal --default-toolchain none
```

The first `cargo` invocation in the workspace will then install the pinned
toolchain plus `rustfmt` and `clippy` on demand.

### System dependencies

The workspace's gRPC crates require `protoc`. On Debian/Ubuntu:

```bash
sudo apt-get install -y protobuf-compiler
```

`protoc` is also installed automatically by the Buildkite Cargo CI script
(`.buildkite/scripts/run-rust-frontend-cargo-ci.sh`) if you want a
reference installation flow.

### Optional helper tools

A few helpers make the local loop faster. Install them once via
[`cargo-binstall`](https://github.com/cargo-bins/cargo-binstall):

```bash
cargo binstall --no-confirm cargo-binstall cargo-nextest cargo-sort
```

- `cargo nextest` — drop-in test runner used by CI (`cargo nextest run`).
- `cargo sort` — keeps `Cargo.toml` sections ordered. CI enforces this.

## Building

### From the repo root, for use with `vllm serve`

`./build_rust.sh` compiles `vllm-rs` and installs it next to the Python
package so `VLLM_USE_RUST_FRONTEND=1 vllm serve` picks it up:

```bash
./build_rust.sh          # release build, recommended for benchmarks
./build_rust.sh --debug  # debug build, faster compile for iteration
```

### From inside the Cargo workspace

When you only need to compile or test the Rust side (no Python integration),
work directly inside `rust/`:

```bash
cargo build --manifest-path rust/Cargo.toml --workspace
```

Most contributors iterate on a single crate. For example, when adding a tool
parser:

```bash
cargo check -p vllm-tool-parser -p vllm-chat
```

## Running tests

CI runs the full workspace under `cargo nextest`. Locally:

```bash
cargo nextest run --manifest-path rust/Cargo.toml --workspace
```

For a single crate or a single test pattern:

```bash
cargo nextest run -p vllm-tool-parser
cargo nextest run -p vllm-chat factory_new_resolves_default_patterns
```

### Roundtrip tests with real HF models

`rust/src/chat/tests/roundtrip.rs` exercises real Hugging Face chat templates
plus the output processor pipeline against a small model matrix. These tests
download tokenizer / template files from Hugging Face on first run and are
serialised per model via `serial_test::file_serial` to avoid concurrent disk
contention. Expect them to take longer than unit tests and to require network
access. Set `HF_HOME` if you want to control the cache location.

### Linters

CI enforces three Rust-specific lint gates:

```bash
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo sort --workspace --check rust
cargo clippy --manifest-path rust/Cargo.toml --workspace \
    --all-targets --all-features --locked -- -D warnings
```

The same checks are wrapped in `.buildkite/scripts/run-rust-frontend-cargo-ci.sh`
under the `style-clippy` mode.

### Pre-commit

vLLM's `pre-commit` config (`.pre-commit-config.yaml`) covers Rust formatting
and Cargo manifest ordering. After `pre-commit install`, the relevant hooks
run automatically on staged Rust files. To run them ad-hoc on a subset of
files:

```bash
pre-commit run --files rust/src/tool-parser/src/json/<your_file>.rs
```

## Adding a tool parser

The Rust tool-parser layer is organised so that most new parsers reuse a
shared streaming core and only contribute a configuration plus tests. The
recurring pattern below mirrors the recently merged JSON-marker parsers
(InternLM2, Phi-4 mini) and the in-flight Granite 4 and ERNIE 4.5 PRs.

### Step 1 — choose the right core

Parsers live by wire-format family under `rust/src/tool-parser/src/`:

| Family | Directory | When to use |
|---|---|---|
| Marker-wrapped JSON object | `json/` | Parser wraps each call as `<marker>{"name":…,"arguments":…}</marker>` — Hermes, Llama 3, Mistral, Qwen XML, InternLM2, Phi-4 mini, etc. |
| DeepSeek DSML | `deepseek_dsml/` | DeepSeek V3.2 / V4 multi-call DSML envelope |
| GLM XML | `glm_xml/` | GLM 4.5 / 4.7 MoE XML format |
| Standalone | top-level `.rs` | Anything not yet abstracted (Kimi K2, Gemma 4, MiniMax M2, Hunyuan v3, …) |

When the new parser fits an existing family, you typically only contribute a
`JsonToolCallConfig` (or equivalent) plus tests; do not duplicate the streaming
state machine.

### Step 2 — write the parser file

For a JSON-marker parser, create `rust/src/tool-parser/src/json/<name>.rs`
shaped like the existing ones:

```rust
use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

const MY_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "My Parser",
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: None,
    name_key: "name",
    arguments_key: &["arguments"],
};

pub struct MyToolParser { inner: JsonToolCallParser }

impl ToolParser for MyToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where Self: Sized + 'static {
        Ok(Box::new(Self { inner: JsonToolCallParser::new(MY_CONFIG) }))
    }
    // Forward parse_into / finish / reset to self.inner.
}
```

If the Python parser sets `request.skip_special_tokens = False` via
`adjust_request(...)`, override `preserve_special_tokens()` to `true`. Otherwise
keep the default (false). This is checked at the chat-output layer when
deciding whether to strip tokenizer special tokens.

### Step 3 — wire up the registrations

Touch five locations, each one a small targeted edit:

1. `rust/src/tool-parser/src/json/mod.rs` — `mod <name>;` and `pub use`.
2. `rust/src/tool-parser/src/lib.rs` — add the type to the public re-export.
3. `rust/src/chat/src/parser/tool/mod.rs`:
   - Add a `pub const <NAME>: &str = "<name>";` in the `names` module
     (matching the Python CLI flag).
   - Call `.register_parser::<MyToolParser>(names::<NAME>)` in
     `ToolParserFactory::new()`, preserving alphabetical order.
   - Optionally call `.register_pattern("<substring>", names::<NAME>)` for
     case-insensitive auto-routing from a model ID. Keep substrings narrow
     enough to avoid collisions with older model versions (see the
     `internlm2` and `granite-4` patterns for examples).
4. `rust/src/chat/src/parser/tool/tests.rs` — extend
   `factory_new_resolves_default_patterns` with at least one positive routing
   case for the new pattern and, when the pattern is version-scoped, one or
   more negative cases that document which related model IDs must NOT route
   here.
5. `rust/src/chat/src/lib.rs` — update the
   `validate_parser_overrides_rejects_unknown_tool_parser` snapshot to insert
   the new name in the alphabetical "choose from" list.

### Step 4 — unit tests in the parser file

Reuse the established shape (see `hermes.rs`, `internlm2.rs`):

- A `parse_complete_*` positive case with raw JSON arguments.
- A `does_not_validate_or_normalize_arguments` case that demonstrates the
  parser passes raw argument text through (no JSON normalisation).
- A `streaming_emits_argument_deltas` case that asserts the observable raw
  argument fragments mid-stream.
- A `streaming_extracts_multiple_tool_calls` case using
  `expect_test::expect![[r#"..."#]].assert_debug_eq(&output)` to snapshot the
  full `ToolParserOutput`. This is the project preference over many
  field-level `assert_eq!`s (see [`rust/AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/rust/AGENTS.md)).
- A `finish_*_incomplete_*` case that snapshots the error report string with
  `expect!` and `thiserror_ext::AsReport`.
- A boolean assertion that pins `preserve_special_tokens()` to the intended
  value.

### Step 5 — design principles

A few project-specific norms that keep parser PRs reviewable:

- **Reuse the core.** Adding a one-off state machine when the existing
  `JsonToolCallParser` (or equivalent) can express the format makes the PR
  harder to review and harder to share fixes across parsers.
- **Do not replicate Python-specific quirks.** Behaviours like `rstrip("\n")`
  after the last tool call, silently swallowing `json.JSONDecodeError`, or
  stripping content after the first `<tool_call>` are Python implementation
  details — the Rust core's uniform behaviour wins. If a model genuinely
  depends on Python-side behaviour, document the divergence in a doc comment
  similar to `internlm2.rs`'s "Differences from Python" / "Known unaddressed
  divergences" sections instead of reaching into the shared core.
- **Land core changes separately.** If a new parser would require new
  options on `JsonToolCallConfig` (multi-arg-key fallback, non-object
  arguments, order-independent header keys, etc.), prefer a small, focused
  core PR first that introduces the knob across all existing parsers, then a
  follow-up parser PR that adopts it.

## Adding a reasoning parser

The reasoning-parser layer (`rust/src/reasoning-parser/`) follows the same
pattern but with a smaller surface: implement the `ReasoningParser` trait,
re-export the type from `lib.rs`, register it in
`rust/src/chat/src/parser/reasoning/mod.rs` (`names` + `register_parser` +
optional `register_pattern`), and update the parser-list snapshot in
`rust/src/chat/src/lib.rs`. Snapshot-style tests still apply.

## CI

Two Rust-focused groups run on PRs that touch `rust/`:

- **Rust Frontend Cargo** — formatting, clippy, `cargo sort`, and the
  workspace test suite. Configured in
  `.buildkite/test_areas/rust_frontend_cargo.yaml`.
- **Rust Frontend E2E** — runs `vllm serve` with
  `VLLM_USE_RUST_FRONTEND=1` against a subset of the Python entrypoint test
  matrix. Configured in `.buildkite/test_areas/rust_frontend.yaml`.

The PR-level `pre-run-check` is a gate, not a lint: it only allows full CI
to run when the PR has the `ready` or `verified` label, or when the author
has at least 4 merged PRs. New contributors should expect this check to
fail until a maintainer applies the label.

## Code owners and where to ask

Code owners for `rust/`, `build_rust.sh`, `rust-toolchain.toml`, and the
Rust Buildkite areas are `@BugenZhao` and `@njhill` (see `.github/CODEOWNERS`).
For scoped work, the
[feature-parity roadmap](https://github.com/vllm-project/vllm/issues/44280)
tracks what is wanted, who has signed up, and which items need a design
discussion first.
