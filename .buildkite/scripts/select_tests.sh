#!/bin/bash
# Select tests to run for a PR using Claude + static import analysis.
# Posts the selection as a PR comment for author review.
#
# Usage:
#   .buildkite/scripts/select_tests.sh <pr_number> [base_branch]
#
# Modes:
#   .buildkite/scripts/select_tests.sh 1234           # select + post comment
#   .buildkite/scripts/select_tests.sh 1234 main      # custom base branch
#   .buildkite/scripts/select_tests.sh --dry-run       # select only, no comment
#
# Requirements:
#   - claude CLI (Anthropic Claude Code)
#   - python3 (for the mapping script)
#   - gh CLI (for posting PR comments, unless --dry-run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
DRY_RUN=false
PR_NUMBER=""
BASE_BRANCH="origin/main"

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
elif [ -n "${1:-}" ]; then
    PR_NUMBER="$1"
fi

if [ -n "${2:-}" ]; then
    BASE_BRANCH="$2"
fi

# ---------------------------------------------------------------------------
# Step 1: Get changed files and diff content
# ---------------------------------------------------------------------------
CHANGED_FILES=$(git diff "${BASE_BRANCH}...HEAD" --name-only)

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed files detected." >&2
    exit 0
fi

FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
echo "Found $FILE_COUNT changed files" >&2

# Capture full diff for Python files (LLM uses this to narrow test selection).
# Truncate at 50KB to control prompt size. Files beyond the cutoff still
# appear in the changed files list — the LLM falls back to mapping for them.
MAX_DIFF_BYTES=51200
FULL_DIFF=$(git diff "${BASE_BRANCH}...HEAD" -- '*.py')
DIFF_BYTES=$(printf '%s' "$FULL_DIFF" | wc -c | tr -d ' ')

if [ "$DIFF_BYTES" -gt "$MAX_DIFF_BYTES" ]; then
    DIFF_CONTENT=$(printf '%s' "$FULL_DIFF" | head -c "$MAX_DIFF_BYTES")
    DIFF_CONTENT="${DIFF_CONTENT}

... (diff truncated at 50KB out of ${DIFF_BYTES} bytes — remaining files use mapping only)"
    echo "Diff truncated from ${DIFF_BYTES} to 50KB" >&2
else
    DIFF_CONTENT="$FULL_DIFF"
fi

# ---------------------------------------------------------------------------
# Step 2: Generate pre-filtered mapping for changed files
# ---------------------------------------------------------------------------
# Convert newline-separated file list to comma-separated for --files flag
FILES_CSV=$(echo "$CHANGED_FILES" | paste -sd ',' -)
echo "Generating pre-filtered import mapping..." >&2
MAPPING=$(python3 "$SCRIPT_DIR/build_test_mapping.py" --files "$FILES_CSV" 2>/dev/null)

# ---------------------------------------------------------------------------
# Step 3: Read static instructions
# ---------------------------------------------------------------------------
INSTRUCTIONS=$(cat "$SCRIPT_DIR/../TEST_SELECTION.md")

# ---------------------------------------------------------------------------
# Step 4: Ask Claude to select tests
# ---------------------------------------------------------------------------
echo "Asking Claude to select tests..." >&2

PROMPT="$(cat <<PROMPTEOF
You are selecting tests for a CI pipeline. Follow the instructions exactly.

## Instructions

${INSTRUCTIONS}

## Candidate Tests (pre-filtered from import analysis for changed files only)

${MAPPING}

## Changed Files in This PR

${CHANGED_FILES}

## Diff Content

${DIFF_CONTENT}

## Your Task

Based on the instructions, the candidate tests, the changed files, and the
diff content above, output the test directories/files to run.
Follow ALL rules in the instructions.
Use the output format specified in the instructions.
PROMPTEOF
)"

SELECTION=$(claude -p --model haiku "$PROMPT" 2>/dev/null)

if [ -z "$SELECTION" ]; then
    echo "Error: Claude returned empty selection." >&2
    exit 1
fi

# Estimate cost (Haiku pricing: $0.25/M input tokens, $1.25/M output tokens)
# Rough estimate: ~4 characters per token
INPUT_CHARS=$(printf '%s' "$PROMPT" | wc -c | tr -d ' ')
OUTPUT_CHARS=$(printf '%s' "$SELECTION" | wc -c | tr -d ' ')
INPUT_TOKENS=$((INPUT_CHARS / 4))
OUTPUT_TOKENS=$((OUTPUT_CHARS / 4))
# Cost in microdollars to avoid floating point
COST_MICROS=$(( (INPUT_TOKENS * 25 + OUTPUT_TOKENS * 125) / 100000 ))
COST_CENTS=$((COST_MICROS / 10))
COST_FRAC=$((COST_MICROS % 10))
echo "Estimated cost: ~\$0.${COST_CENTS}${COST_FRAC} (${INPUT_TOKENS} input + ${OUTPUT_TOKENS} output tokens)" >&2

# ---------------------------------------------------------------------------
# Step 5: Parse reasoning and test list, build the PR comment
# ---------------------------------------------------------------------------

# Split output on "---" separator into reasoning and test list
REASONING=$(echo "$SELECTION" | sed -n '1,/^---$/p' | sed '$d')
TEST_LIST=$(echo "$SELECTION" | sed -n '/^---$/,$ p' | sed '1d')

# Filter test list to valid "path | reason" lines
CLEAN_SELECTION=$(echo "$TEST_LIST" | grep -E '^[[:space:]]*[a-zA-Z0-9_/.-]+ *\|' || true)

if [ -z "$CLEAN_SELECTION" ]; then
    echo "Warning: Could not parse Claude's output. Raw output:" >&2
    echo "$SELECTION" >&2
    exit 1
fi

# Check for NONE case
IS_NONE=$(echo "$CLEAN_SELECTION" | grep -ic '^[[:space:]]*NONE ' || true)

if [ "$IS_NONE" -gt 0 ]; then
    NONE_REASON=$(echo "$CLEAN_SELECTION" | head -1 | cut -d'|' -f2 | xargs)
    TARGET_COUNT=0
else
    TARGET_COUNT=$(echo "$CLEAN_SELECTION" | wc -l | tr -d ' ')
fi

# Build the comment body
COMMENT_BODY="## Suggested Test Selection

**${FILE_COUNT} files changed"

if [ "$IS_NONE" -gt 0 ]; then
    COMMENT_BODY="${COMMENT_BODY}, no tests selected.**

### Reasoning

${REASONING}

**Result:** ${NONE_REASON}

<details><summary>Changed files</summary>

\`\`\`
${CHANGED_FILES}
\`\`\`

</details>"
else
    COMMENT_BODY="${COMMENT_BODY}, ${TARGET_COUNT} test targets selected.**

### Reasoning

${REASONING}

### Selected tests

| Test target | Reason |
|---|---|
$(echo "$CLEAN_SELECTION" | while IFS='|' read -r target reason; do
    target=$(echo "$target" | xargs)
    reason=$(echo "$reason" | xargs)
    if [ -n "$target" ]; then
        echo "| \`${target}\` | ${reason} |"
    fi
done)

<details><summary>Changed files (${FILE_COUNT})</summary>

\`\`\`
${CHANGED_FILES}
\`\`\`

</details>"
fi

COMMENT_BODY="${COMMENT_BODY}

### Review

If this selection looks wrong, reply with what should change:
- **Missing tests**: \"Also run tests/distributed/ because this change affects X\"
- **Unnecessary tests**: \"tests/compile/ is not relevant here because Y\"

If it looks right, no action needed.

---
*Auto-generated by test selection script.*"

# ---------------------------------------------------------------------------
# Step 6: Post or print
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = true ]; then
    echo "$COMMENT_BODY"
else
    if [ -z "$PR_NUMBER" ]; then
        echo "Error: PR number required. Use --dry-run for local testing." >&2
        exit 1
    fi

    # Delete previous test selection comment if it exists (avoid stacking)
    PREVIOUS_COMMENT_ID=$(gh api \
        "repos/{owner}/{repo}/issues/${PR_NUMBER}/comments" \
        --jq '.[] | select(.body | startswith("## Suggested Test Selection")) | .id' \
        2>/dev/null | head -1)

    if [ -n "$PREVIOUS_COMMENT_ID" ]; then
        echo "Deleting previous test selection comment..." >&2
        gh api -X DELETE \
            "repos/{owner}/{repo}/issues/comments/${PREVIOUS_COMMENT_ID}" \
            2>/dev/null || true
    fi

    # Post new comment
    gh pr comment "$PR_NUMBER" --body "$COMMENT_BODY"
    echo "Comment posted to PR #${PR_NUMBER}" >&2
fi

# Also output just the test paths to stdout (for CI pipeline use later)
if [ "$IS_NONE" -eq 0 ]; then
    echo "$CLEAN_SELECTION" | while IFS='|' read -r target _reason; do
        target=$(echo "$target" | xargs)
        if [ -n "$target" ]; then
            echo "$target"
        fi
    done
fi
