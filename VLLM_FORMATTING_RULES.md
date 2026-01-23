# vLLM Pre-Commit Formatting Rules

This document summarizes the formatting rules enforced by vLLM's pre-commit CI.

## C++ Files (.hpp, .cpp)

### clang-format Configuration

vLLM uses `.clang-format` with these key settings:

```yaml
BasedOnStyle: Google
IndentWidth: 2
ColumnLimit: 80
IndentPPDirectives: BeforeHash
```

The `IndentPPDirectives: BeforeHash` is **critical** - it means preprocessor
directives get indented based on their code nesting level.

### clang-format Rules

1. **Indentation**: 2 spaces everywhere (function bodies, conditionals, etc.)
   ```cpp
   // CORRECT
   static inline int foo() {
     if (condition) {
       return 1;
     }
     return 0;
   }
   ```

2. **Preprocessor Directives Inside Function Bodies**: Need 2-space indent!
   ```cpp
   // CORRECT - inside TORCH_LIBRARY_EXPAND function body
   TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
     // Regular code at 2-space indent
     ops.def("something");

     // Preprocessor inside function body also gets 2-space!
     #ifndef VLLM_POWER8_COMPAT
     ops.def("another_thing");
     #endif
   }

   // WRONG - preprocessor at column 0 inside function body
   TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
   #ifndef VLLM_POWER8_COMPAT  // <-- NO! Should be "  #ifndef"
     ops.def("another_thing");
   #endif                       // <-- NO! Should be "  #endif"
   }
   ```

3. **Nested Preprocessor Directives**: Each nesting level adds 2 spaces
   ```cpp
   // CORRECT - at top level
   #if defined(__POWER8_VECTOR__)
     #if defined(TORCH_VERSION_MAJOR)
       #define VLLM_POWER8_COMPAT 1
     #endif
   #endif

   // Inside an #ifdef block, includes are indented too:
   #ifdef GGML_USE_MASS
     #include <massv.h>
     #if defined(__IBMC__)
       #include <mass_simd.h>  // 4-space: nested inside both blocks
     #endif
   #endif
   ```

4. **Long `#if` Conditions**: Use backslash continuation
   ```cpp
   // CORRECT
   #if TORCH_VERSION_MAJOR < 2 || \
       (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)
   ```

5. **`#endif` Comments**: Single space after `#endif`
   ```cpp
   // CORRECT
   #endif /* __linux__ */

   // WRONG
   #endif  /* __linux__ */   // double space!
   ```

6. **`#include` Inside `#ifdef`**: Need 2-space indent
   ```cpp
   // CORRECT
   #ifdef __linux__
     #include <numa.h>
   #endif
   ```

## Markdown Files (.md)

### mdformat / markdownlint Rules

1. **Code Blocks**: Blank lines BEFORE and AFTER every fenced block

   CORRECT: Add blank line before and after the triple-backtick block.
   Text → blank line → \`\`\`python → code → \`\`\` → blank line → text

2. **Headings Followed by Lists**: Blank line BETWEEN heading and list
   ```markdown
   <!-- CORRECT -->
   ### What Works

   - Item 1
   - Item 2

   <!-- WRONG -->
   ### What Works
   - Item 1
   ```

3. **Headings Surrounded by Blank Lines**: Before AND after
   ```markdown
   <!-- CORRECT -->
   Some content.

   ## Heading

   More content.
   ```

## Python Files (.py)

### Ruff Linter Rules

1. **Blank Lines Between Functions**: Even stub functions need separation
   ```python
   # CORRECT
   def get_num_threads():
       return 1


   def set_num_threads(n):
       pass
   ```

2. **Multi-Line Function Bodies**: Avoid one-liner functions with pass
   ```python
   # CORRECT
   def set_num_threads(n):
       pass

   # Ruff may want this expanded
   ```

3. **Import Grouping**: Standard library, third-party, then local imports with blank lines between groups

4. **Line Length**: E501 - lines should not exceed 88 characters (or configured limit)

## DCO (Developer Certificate of Origin)

Every commit must have:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Add automatically with: `git commit -s`

## Quick Checks

### Check C++ Formatting Locally

```bash
# Run clang-format on your files
clang-format -i csrc/cpu/your_file.cpp
clang-format -i csrc/cpu/your_file.hpp
```

### Check Python Formatting Locally

```bash
# Install ruff
pip install ruff

# Check files
ruff check vllm/your_file.py

# Auto-fix
ruff check --fix vllm/your_file.py
```

### Check Markdown Locally

```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Check file
markdownlint POWER8_VLLM_BUILD_GUIDE.md
```

## Common Pitfalls

1. **Copy-pasting code with 4-space indent** - vLLM uses 2-space
2. **Forgetting blank lines around code blocks** in markdown
3. **Nested preprocessor directives** - each level needs 2-space indent
4. **Headings immediately before lists** - need blank line between
5. **`#endif` with double spaces** before comments
6. **Preprocessor at column 0 inside functions** - needs 2-space indent!
7. **Short functions with multi-line bodies** - clang-format may want one-liners:
   ```cpp
   // clang-format prefers this:
   static inline __vector float vsexp4(__vector float v) { return vec_exp(v); }

   // NOT this (for simple one-statement functions):
   static inline __vector float vsexp4(__vector float v) {
       return vec_exp(v);
   }
   ```

## Key Insight: Nesting Determines Indent

The critical rule is: **preprocessor directive indent depends on NESTING LEVEL, not function context**.

### NOT Nested (Column 0)

```cpp
// At file scope OR at function level but NOT inside another #if
TORCH_LIBRARY_EXPAND(ops) {
  ops.def("foo");

// Rotary embedding comment
#ifndef VLLM_POWER8_COMPAT       // <-- Column 0 (NOT nested in another #if)
  ops.def("rotary_embedding...");
#endif                           // <-- Column 0
}
```

### Nested (2-Space Indent)

```cpp
#if defined(__AVX512F__)         // <-- Column 0 (top level)
  #ifndef VLLM_POWER8_COMPAT     // <-- 2-space (INSIDE the __AVX512F__ block)
  ops.def("avx512_thing");
  #endif                         // <-- 2-space
#endif                           // <-- Column 0
```

## Template Spacing

clang-format requires space after `template` keyword:

```cpp
// CORRECT
template <typename V, typename T>
static inline void foo() {}

// WRONG
template<typename V, typename T>   // <-- Missing space!
static inline void foo() {}
```

## Union Formatting

Multi-line unions with braces on separate lines:

```cpp
// CORRECT
union {
  V v;
  char c[16];
} u;

// WRONG
union { V v; char c[16]; } u;
```

## Learned the Hard Way

These rules were discovered through 8+ rounds of pre-commit CI failures on PR #31512:

| Commit | Issue Fixed |
|--------|-------------|
| 0e583106e | `#endif` single space, basic 2-space indent |
| 19d5f9ed7 | hpp function bodies 2-space, nested `#if` indent |
| e447d3b5d | Remaining `#endif` lines for POWER8_COMPAT blocks |
| 86cb39af9 | Markdown code block blank lines |
| eef1537c5 | Headings need blank lines before lists |
| deeed74f7 | Macro formatting, tried column 0 for `#ifndef` |
| 7195a8bc6 | `#ifndef`/`#endif` need 2-space inside function bodies! |
| e72da462f | Non-nested blocks at column 0, nested at 2-space |
| 7cda78156 | Template spacing, union formatting, duplicate macro cleanup |
| 59d2e4db9 | List numbering, nested code block example |
| 414185df0 | Heading trailing colons, blank lines below headings |
| da2ae7362 | Blank lines before code blocks |
| 50cefe4d9 | Table trailing pipe |
| d1d2a7be9 | Heading-to-code-block spacing |
| 30d69806b | Quick Checks section spacing |
| c9fc20010 | Code block language specifier (MD040) - **PASSED!** |
