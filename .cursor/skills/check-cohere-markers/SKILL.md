---
name: check-cohere-markers
description: Check that cohere markers are correctly applied to all modifications in existing upstream files. Compares current file contents directly against a confirmed upstream tag/commit (vllm-project/vllm). Use proactively before pushing changes or during code review to ensure all custom modifications to upstream files are annotated with cohere comments.
---

# Check Cohere Markers

Verify that all modifications to existing upstream files are properly annotated with cohere markers. New files and folders added by the custom branch do NOT need markers — only changes to files that exist in **upstream** (`git@github.com:vllm-project/vllm.git`) require them.

## When to Use

- Before pushing local commits to remote
- During code review
- After resolving rebase conflicts
- As a pre-push sanity check
- When invoked by the `rebase-assistant` skill (via `minimize-upstream-diff` in Step 3)

## Workflow

### Step 1: Collect Modified Upstream Files

Follow the upstream file collection workflow in
[`../_shared/upstream-file-collection.md`](../_shared/upstream-file-collection.md)
(Steps A through C) to determine `UPSTREAM_REF` and collect the list of
in-scope files (modified upstream files that need marker checking).

### Step 2: Determine Comment Syntax per File

Use the correct comment syntax based on file extension:

| File Type | Extensions | Single-line | Block start | Block end |
|-----------|-----------|-------------|-------------|-----------|
| Python | `.py` | `# cohere` | `# cohere start` | `# cohere end` |
| C/C++/CUDA | `.c`, `.cpp`, `.h`, `.hpp`, `.cu`, `.cuh` | `// cohere` | `// cohere start` | `// cohere end` |
| Shell | `.sh`, `.bash` | `# cohere` | `# cohere start` | `# cohere end` |
| YAML | `.yaml`, `.yml` | `# cohere` | `# cohere start` | `# cohere end` |
| CMake | `CMakeLists.txt`, `.cmake` | `# cohere` | `# cohere start` | `# cohere end` |
| Dockerfile | `Dockerfile*` | `# cohere` | `# cohere start` | `# cohere end` |
| JavaScript/TypeScript | `.js`, `.ts`, `.jsx`, `.tsx` | `// cohere` | `// cohere start` | `// cohere end` |
| Markdown | `.md` | `<!-- cohere -->` | `<!-- cohere start -->` | `<!-- cohere end -->` |
| HTML/Jinja | `.html`, `.jinja`, `.jinja2` | `<!-- cohere -->` | `<!-- cohere start -->` | `<!-- cohere end -->` |
| TOML | `.toml` | `# cohere` | `# cohere start` | `# cohere end` |
| INI/Config | `.ini`, `.cfg`, `.conf` | `# cohere` | `# cohere start` | `# cohere end` |

For unlisted file types, infer from the file's language or prompt the user.

### Step 3: Check Each Modified Upstream File

For each file that exists in upstream and was modified:

1. **Get the full content diff** for that file against upstream:
   ```bash
   git diff "$UPSTREAM_REF" -- <file>
   ```

2. **Read the current file content** to see the full context around changes.

3. **For each changed hunk, check for cohere markers**:
   - Every added or modified line (or contiguous block) should have a marker
   - **Single-line change**: The line itself should end with the appropriate comment (e.g., `# cohere`)
   - **Contiguous block of changes**: The block should be wrapped with start/end markers (e.g., `# cohere start` before, `# cohere end` after)
   - **New function/class/section within the file**: Wrapped with start/end markers

4. **Validate marker pairing**: Every `cohere start` must have a matching `cohere end` (and vice versa). Flag unmatched markers.

5. **Validate marker syntax**: Ensure the comment syntax matches the file type.

### Step 4: Report Findings

Produce a report with three sections:

#### 1. Files with correct markers (PASS)
List files where all changes are properly annotated.

#### 2. Files with missing or incorrect markers (FAIL)
For each file:
- Show the file path
- Show each unmarked change with line numbers and surrounding context
- Suggest the exact fix (which marker to add and where)

#### 3. Files that don't need markers (SKIP)
List new files/folders (status `A`) that were correctly excluded.

### Step 5: Fix Missing Markers (Interactive)

If there are missing markers:

1. **Show the summary** of missing markers to the user
2. **Ask user**: "Would you like me to add the missing markers? (yes/no/review-each)"
   - **yes**: Add all missing markers automatically
   - **no**: Leave as-is, user will fix manually
   - **review-each**: Go through each missing marker one at a time, asking the user to confirm before adding

When adding markers:
- For isolated single-line changes, append the inline comment
- For blocks of 2+ contiguous changed lines, wrap with start/end markers
- Place `cohere start` on the line immediately before the first changed line
- Place `cohere end` on the line immediately after the last changed line

After adding markers, re-run the check (Step 3) to confirm everything is covered.

### Step 6: Final Summary

After all fixes:
- Show count of files checked, passed, failed, skipped
- List any remaining issues
- If all clean: "All cohere markers are correctly applied."

## Edge Cases

- **Mixed changes**: A file with both modified and newly-added sections — only the modifications to code that existed upstream need markers; entirely new sections within the file still need markers since the file itself is an upstream file.
- **Moved/renamed files** (`R` status): If content also changed, treat changed portions like `M` files.
- **Binary files**: Skip — markers don't apply to binary content.
- **Empty diffs**: Skip files where git reports a status change but no actual content diff.
- **Nested markers**: If a `cohere start`/`cohere end` block already exists and a change is made inside it, no additional marker is needed for lines within that block.

## Usage Examples

**Check current branch content against upstream (will prompt to confirm the ref):**
```
Use check-cohere-markers to verify my changes
```

**Provide an upstream ref upfront to skip the suggestion step (confirmation still required):**
```
Use check-cohere-markers against v0.14.0
```

**Fix missing markers automatically:**
```
Use check-cohere-markers and fix any missing markers
```
