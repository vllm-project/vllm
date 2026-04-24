#!/bin/bash
# Helper script for resolving rebase conflicts and training git rerere
# This script helps you resolve conflicts locally and ensures rerere learns the resolutions

set -e

UPSTREAM_REF="${1:-vllm-upstream/main}"
TARGET_BRANCH="${2:-cohere-synced}"

# Infer source branch from target branch (remove -synced suffix)
SOURCE_BRANCH="${TARGET_BRANCH%-synced}"
SQUASHED_BRANCH="${SOURCE_BRANCH}-squashed"

echo "=== Git Rerere Conflict Resolution Helper ==="
echo "Upstream: $UPSTREAM_REF"
echo "Source branch: $SQUASHED_BRANCH"
echo "Target branch: $TARGET_BRANCH"
echo ""

# Enable rerere
git config rerere.enabled true
git config rerere.autoUpdate true

# Download existing rerere cache from GitHub first
echo "Checking for existing rerere cache on GitHub..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/download-rerere-cache.sh" || echo "⚠️  Could not download cache (continuing anyway)"
echo ""

# Add upstream if needed
if ! git remote | grep -q "^vllm-upstream$"; then
    echo "Adding upstream remote..."
    git remote add vllm-upstream https://github.com/vllm-project/vllm.git
fi

# Fetch latest
echo "Fetching upstream and squashed branch..."
git fetch vllm-upstream
git fetch origin $SQUASHED_BRANCH

# Validate the upstream ref format
# Fail fast if someone passes remote/SHA (invalid format)
if [[ "$UPSTREAM_REF" =~ ^[a-z-]+/([0-9a-f]{7,40})$ ]]; then
    echo "❌ Error: Invalid upstream ref format: $UPSTREAM_REF"
    echo ""
    echo "You passed what looks like a remote name with a commit SHA."
    echo "This is not valid git syntax."
    echo ""
    echo "Valid formats:"
    echo "  - Branch: vllm-upstream/main"
    echo "  - Tag: v0.6.0"
    echo "  - Commit SHA: 1da94e673c257373280026f75ceb4effac80e892"
    echo ""
    echo "NOT valid:"
    echo "  - vllm-upstream/1da94e673c... ❌"
    exit 1
fi

# Start rebase
echo "Starting rebase from $SQUASHED_BRANCH onto $UPSTREAM_REF..."
git checkout -B $TARGET_BRANCH origin/$SQUASHED_BRANCH

# Call shared rebase logic
"$SCRIPT_DIR/rebase-with-rerere.sh" "$UPSTREAM_REF" "$TARGET_BRANCH"
REBASE_RESULT=$?

if [ $REBASE_RESULT -eq 0 ]; then
    echo ""
    echo "Next steps:"
    echo "1. git push -f origin $TARGET_BRANCH"
    echo "2. .github/scripts/upload-rerere-cache.sh  (to share with GitHub Actions)"
elif [ $REBASE_RESULT -eq 1 ]; then
    echo ""
    echo "Rerere status:"
    git rerere status || echo "No rerere resolutions available"
    echo ""
    echo "================================"
    echo "Next steps:"
    echo "1. Resolve conflicts in your editor"
    echo "2. git add <resolved-files>"
    echo "3. git rebase --continue"
    echo "4. Repeat until rebase completes"
    echo "5. git push -f origin $TARGET_BRANCH"
    echo "6. .github/scripts/upload-rerere-cache.sh  (to share with GitHub Actions)"
    echo ""
    echo "Rerere will automatically record your resolutions!"
    echo "================================"
    exit 1
else
    echo "❌ Error occurred during rebase"
    exit 2
fi
