# Alternative Frontend to vLLM Engine in Rust

This project aims to implement an alternative frontend to the vLLM Engine in Rust, providing a more efficient and robust interface for interacting with the engine. Currently it's still in the very early stage and is actively evolving.

# Coding Styles

- Always use workspace dependencies for Cargo crates.
- Prefer splitting code into multiple smaller modules and files for better organization and readability, rather than putting everything in a single file.
- When refactoring or reconstructing code, always preserve the original comments and documentation VERBATIM, if applicable.
- If not specified, default to writing concise Rust documentation and comments that match the style of the existing codebase when generating code.
- When migrating code from Python or any other language, preserve the original documentation comments whenever they still make sense in the Rust code.
- Although you might be asked to only implement or migrate minimal functionality at the beginning, you should still leave necessary `TODO` comments in the code for the future improvements of the lacked features, so that it's easier for the next iteration to build upon the existing codebase.
- Rust error handling:
  - Never call `to_string()` directly on an error value.
  - Use `ToReportString` or `AsReport` by `thiserror-ext` instead.
- Since the project is still in early stage, it's fine to break API and make non-backwards-compatible changes as needed.

# Testing

- Prefer snapshot testing with the `expect-test` crate over writing multiple `assert_eq!` statements on individual fields. Use `expect_test::expect![[...]].assert_debug_eq(...)` to snapshot the `Debug` output of the entire struct.
  - Write `expect![[""]]` as a placeholder first, then run `UPDATE_EXPECT=1 cargo test` to auto-fill the snapshot content.
  - For values containing non-deterministic data (e.g., UUIDs), set them to a fixed value like `"<placeholder>"` before snapshotting.
