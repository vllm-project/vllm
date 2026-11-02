# Alternative Frontend to vLLM Engine in Rust

This project implements an alternative, Rust-native frontend to the vLLM Engine. It is still experimental and actively evolving. The Python frontend is the behavioral reference for the OpenAI-compatible API; see [Python frontend parity](#python-frontend-parity).

## Crate ownership

Requests flow `HTTP/gRPC DTO → chat or text request → engine request`. Dependencies point downward: lower crates know nothing about the crates above them. Put a rule in the crate that owns its semantics, even if only one protocol triggers it today.

| Crate | Owns |
| --- | --- |
| `vllm-server` | HTTP and gRPC entry points, wire DTOs and wire-level validation, lowering DTOs into chat or text requests, building responses, error mapping |
| `vllm-proto` | gRPC protobuf definitions |
| `vllm-chat` | Chat template rendering, multimodal inputs, structured assistant events, driving reasoning and tool parsers, response aggregation |
| `vllm-parser` | Streaming reasoning and tool-call parsers and output grammars; protocol recognition only |
| `vllm-text` | Prompt text handling, tokenizer and model loading, incremental detokenization, lowering text requests into engine generate requests |
| `vllm-tokenizer` | Tokenizer backends and incremental decoding |
| `vllm-llm` | Token-in/token-out facade over the engine client, with request timing, metrics, and stats |
| `vllm-engine-core-client` | ZMQ transport and the MessagePack protocol to the engine core |
| `vllm-cmd` (`vllm-rs`) | CLI entry points and process wiring |

The remaining crates are tooling and infrastructure (`vllm-metrics`, `vllm-tracing`, `vllm-build-info`, `vllm-managed-engine`, `vllm-mock-engine`, `vllm-bench`). Check `Cargo.toml` for the current dependency graph before adding a cross-crate dependency.

## Design principles

1. **Trace the actual path before changing it.** Explain a bug with a concrete input and output, then ask whether the structure that produces it is needed at all; removing an unneeded task or branch often beats patching it.
2. **Use strong types for known semantics.** Enums, structs, borrowed views, and typed errors replace strings, flags, and loose JSON for things like modalities, sampling modes, tool choice, and response format. Make the meaning of an absent `Option`, an empty collection, and the point where defaults apply visible in the code. Stop where the type costs more than it buys: a newtype needs a concrete constraint, such as rejecting invalid values at parse time or preventing invalid combinations.
3. **Give each type and parameter one owner.** A type carries only the fields its own consumer reads; when splitting fields out of a request type, use `#[serde(flatten)]` so the wire shape stays unchanged. Trait methods take only what implementations read. A parameter every caller passes with the same value belongs inside the type that owns it.
4. **Prefer a small library over a framework.** Extract shared semantics as a small trait plus a few building blocks, not a config-driven engine with knobs for each consumer. Before adding a flag and new branches, try expressing the new mode as a value in the existing data path. A little duplication across separate responsibilities is fine.
5. **Fix configuration at construction; pass events to methods.** Policies, clients, and model config are fixed when the owner is built, ideally once at model load; requests, deltas, and chunks are method arguments. Parsers and streams keep construction, per-chunk updates, and finishing as separate steps. Register a resource and its cleanup together, and walk the success, error, cancellation, and drop paths. Where async cancellation opens a window, hold an RAII guard before the first relevant `.await`.
6. **Tie every defensive check to a reachable input.** For each check, name the input that reaches it, what goes wrong without it, and which code owns it: the code that consumes the input, checking only what that consumer reads. Normalize first, then validate canonical values in one place. A branch that silently narrows output is worse than no branch.
7. **Reject behavior that is not implemented.** A request field or CLI flag set to a non-default value whose behavior the Rust frontend does not implement returns a 400 or fails at startup; it is not warned about and ignored. Flags that have no meaning in the Rust frontend are accepted as documented no-ops (search for `Noop`).

    ```rust
    if request.sampling_params.include_stop_str_in_output == Some(true) {
        bail_invalid_request!(
            param = "include_stop_str_in_output",
            "include_stop_str_in_output is not supported."
        );
    }
    ```

8. **Keep compatibility code to real consumers.** Internal Rust APIs can change freely: migrate the callers and delete the old path in the same PR. A compatibility layer on a wire contract needs an actual consumer, such as independently versioned gRPC clients or a rolling upgrade, and stays narrow at the wire boundary.

## Python frontend parity

- Match the Python frontend's wire behavior: serialization, omitted fields versus `null`, defaults, error status codes, and SSE frame order. Check the actual output, not the Python source alone.
- When porting from Python, compare the port against the original line by line. Each deviation in observable behavior needs a basis (the wire contract, the official chat template or checkpoint files, or a reference implementation) and a note in the code or the PR description.
- Port the contract, not incidental behavior. Do not reproduce a Python workaround or an edge-case quirk that no client depends on; do document the divergence.
- Parity is in behavior, not structure. Rust types, ownership, and library usage stay idiomatic, and wire parity does not require Python's field placement.
- When a behavior fix changes a contract both frontends share, keep them consistent; a small fix in the Python owner that the change directly needs can land in the same PR.

## Parsers

- A parser recognizes its protocol and emits ordered events, keeping only the state parsing needs. Response aggregation, assigning tool-call IDs, output policy, and the request lifecycle belong to the callers in `vllm-chat` and `vllm-server`; a parser reports a tool-call ID only when the model generated it.
- Non-streaming output consumes the final state of the streaming parser, so both modes have one source for content and IDs.
- A model whose behavior differs gets its own parser type: a thin wrapper that delegates generic parsing and overrides one extension point. Register one type under several names only when the behavior is identical. Model names stay out of generic code paths and error messages.
- Derive the initial state from the real rendered prompt, such as a reasoning block the chat template already opened. Where the parser receives decoded tokens, match special markers by token identity rather than by spelling alone (search for `Marker::special`), and preserve whitespace and chunk boundaries.
- Set the supported input shapes by what official chat templates and real generation produce. Handcrafted malformed inputs are fine for robustness tests but are not regressions without a real trigger.
- When writing parsers with `winnow`:
    - Prefer a declarative parser shape over imperative step-by-step parsing, as long as it's more readable and maintainable.
    - Prefer tuple-based parser composition over calling `parse_next` one parser at a time.
    - Prefer built-in combinators and token parsers before adding local helpers.
    - Add short documentation comments like `Parse a ..` to all local parser/combinator functions.
    - Reuse existing utilities from `utils` module as much as possible, and add new ones there if needed.

## Coding Styles

- Always use workspace dependencies for Cargo crates.
- Split code into modules by what they own. Every new file, helper, wrapper, and parameter needs a responsibility you can name; inline helpers that only forward a call, and widen visibility only as far as actual callers require.
- When refactoring or reconstructing code, always preserve the original comments and documentation VERBATIM, if applicable.
- If not specified, default to writing concise Rust documentation and comments that match the style of the existing codebase when generating code. Comments explain contracts, non-obvious invariants, and reasons; they do not narrate the code.
- When migrating code from Python or any other language, preserve the original documentation comments whenever they still make sense in the Rust code.
- Although you might be asked to only implement or migrate minimal functionality at the beginning, you should still leave necessary `TODO` comments in the code for the future improvements of the lacked features, so that it's easier for the next iteration to build upon the existing codebase.
- Reuse the vocabulary already in the codebase and the protocol. Names state roles and effects, and a rename propagates through locals, tests, comments, and docs.
- Derive serialization with Serde attributes instead of hand-writing it; absorb schema variants with `#[serde(alias)]` instead of parallel fields. Choose `Display` or `Debug` by meaning.
- Rust error handling:
    - Error types describe the categories their layer actually produces. Use `#[from]` and `?` for one-to-one conversions between typed errors, and an explicit `map_err` where a general type carries several meanings.
    - Never call `to_string()` directly on an error value.
    - Use `ToReportString` or `AsReport` by `thiserror-ext` instead.
    - For `Error` variants that are primarily free-form text, prefer a struct variant with a `message: String` field. `thiserror_ext::Macro` will auto-derive `foo!(...)` and `bail_foo!(...)` helper macros from that shape.
        - Use `foo!(...)` when you need to construct an error value inside an expression, such as `Err(foo!(...))`, `.ok_or_else(|| foo!(...))`, or `Err::<(), _>(foo!(...))?`.
        - Use `bail_foo!(...)` only in statement positions where you want to exit the current `Result`-returning function immediately. Prefer it over `return Err(foo!(...))` in those cases.
        - If a variant has extra structured fields, prefer the generated macro form `foo!(field = value, "message")` rather than manually writing `Error::Foo { ... }`.
- Currently the project is only targeting Unix-like platforms, so it's fine to use Unix-specific APIs without extra compatibility layers like `cfg(unix)`
- Encode a rule that can be checked mechanically in `clippy.toml` or `deny.toml` rather than here; for example, `clippy.toml` disallows the locking Prometheus histogram.

## Testing

- For behavior that crosses crates, assert on the final consumer's output.
- A useful test fails on a naive or broken implementation and keeps passing when only the internals are restructured. Prove a regression test by reverting the fix and watching it fail.
- Prefer snapshot testing with the `expect-test` crate over writing multiple `assert_eq!` statements on individual fields. Use `expect_test::expect![[...]].assert_debug_eq(...)` to snapshot the `Debug` output of the entire struct.
    - Write `expect![[""]]` as a placeholder first, then run `UPDATE_EXPECT=1 cargo test` to auto-fill the snapshot content.
    - For values containing non-deterministic data (e.g., UUIDs), set them to a fixed value like `"<placeholder>"` before snapshotting.
    - Use `expect_file!` for large outputs, and read every generated snapshot before committing it. Identical snapshots across variants that should differ mean the test misses the distinction.
- In tests, avoid hand-writing full request struct literals when only a few fields matter. Prefer test fixtures such as `for_test()` with struct update syntax, so newly added fields do not force mechanical edits across many tests.
- Prefer deterministic synchronization in async and integration tests, such as channels, barriers, explicit handshakes, or observable state transitions, instead of `sleep`-based timing assumptions.
    - Use `sleep` only as a last resort when there is no better observable synchronization point.
- A pure refactor usually needs no new unit tests. Do not skip tests silently; an explicit skip is fine when the environment lacks a precondition.
- Always run test with `cargo nextest run` instead of `cargo test`, if available, as it's much faster.

## Validation

Match the evidence to the claim, and say which layer it reached:

- A refactor preserves behavior: compilation and the relevant existing tests.
- A renderer or serializer matches Python: the final wire output, rendered prompt, or complete token IDs, including defaults and edge cases.
- HTTP or gRPC handling and error mapping: the in-process route and gRPC tests against the scripted engine in `vllm_engine_core_client::mock_engine`. A live server is needed only for claims about deployment and process wiring.
- Load, soak, or transport throughput: the `vllm-mock-engine` binary. It emits random tokens, so it cannot establish anything about model output.
- Model output, reasoning, tool calls, or structured output: a real checkpoint, with a paired baseline when quality is the question.

For performance claims, measure with `vllm-bench`, count failed requests, and report wall time and CPU time separately when the code runs work in parallel.

## Before opening a PR

- Keep mechanical moves in separate commits from semantic changes.
- In the PR description, list any intentional deviation from this guide or from the Python frontend's behavior, with the reason.
