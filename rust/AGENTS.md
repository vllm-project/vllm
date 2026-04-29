# Alternative Frontend to vLLM Engine in Rust

This project aims to implement an alternative frontend to the vLLM Engine in Rust, providing a more efficient and robust interface for interacting with the engine. Currently it's still in the very early stage and is actively evolving.

# Coding Styles

- Always use workspace dependencies for Cargo crates.
- Prefer splitting code into multiple smaller modules and files for better organization and readability, rather than putting everything in a single file.
- When refactoring or reconstructing code, always preserve the original comments and documentation VERBATIM, if applicable.
- If not specified, default to writing concise Rust documentation and comments that match the style of the existing codebase when generating code.
- When migrating code from Python or any other language, preserve the original documentation comments whenever they still make sense in the Rust code.
- Although you might be asked to only implement or migrate minimal functionality at the beginning, you should still leave necessary `TODO` comments in the code for the future improvements of the lacked features, so that it's easier for the next iteration to build upon the existing codebase.
- When writing parsers with `winnow`:
  - Prefer a declarative parser shape over imperative step-by-step parsing, as long as it's more readable and maintainable.
  - Prefer tuple-based parser composition over calling `parse_next` one parser at a time.
  - Prefer built-in combinators and token parsers before adding local helpers.
  - Add short documentation comments like `Parse a ..` to all local parser/combinator functions.
- Rust error handling:
  - Never call `to_string()` directly on an error value.
  - Use `ToReportString` or `AsReport` by `thiserror-ext` instead.
  - For `Error` variants that are primarily free-form text, prefer a struct variant with a `message: String` field. `thiserror_ext::Macro` will auto-derive `foo!(...)` and `bail_foo!(...)` helper macros from that shape.
    - Use `foo!(...)` when you need to construct an error value inside an expression, such as `Err(foo!(...))`, `.ok_or_else(|| foo!(...))`, or `Err::<(), _>(foo!(...))?`.
    - Use `bail_foo!(...)` only in statement positions where you want to exit the current `Result`-returning function immediately. Prefer it over `return Err(foo!(...))` in those cases.
    - If a variant has extra structured fields, prefer the generated macro form `foo!(field = value, "message")` rather than manually writing `Error::Foo { ... }`.
- Since the project is still in early stage, it's fine to break API and make non-backwards-compatible changes as needed.
- Currently the project is only targeting Unix-like platforms, so it's fine to use Unix-specific APIs without extra compatibility layers like `cfg(unix)`

# Testing

- Prefer snapshot testing with the `expect-test` crate over writing multiple `assert_eq!` statements on individual fields. Use `expect_test::expect![[...]].assert_debug_eq(...)` to snapshot the `Debug` output of the entire struct.
  - Write `expect![[""]]` as a placeholder first, then run `UPDATE_EXPECT=1 cargo test` to auto-fill the snapshot content.
  - For values containing non-deterministic data (e.g., UUIDs), set them to a fixed value like `"<placeholder>"` before snapshotting.
- In tests, avoid hand-writing full request struct literals when only a few fields matter. Prefer test fixtures such as `for_test()` with struct update syntax, so newly added fields do not force mechanical edits across many tests.
- Prefer deterministic synchronization in async and integration tests, such as channels, barriers, explicit handshakes, or observable state transitions, instead of `sleep`-based timing assumptions.
  - Use `sleep` only as a last resort when there is no better observable synchronization point.
- Always run test with `cargo nextest run` instead of `cargo test`, if available, as it's much faster.
