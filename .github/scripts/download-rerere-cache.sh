#!/bin/bash
# Download rerere cache from GitHub to local machine

set -e

echo "=== Downloading rerere cache from GitHub ==="

# Enable rerere
git config rerere.enabled true
git config rerere.autoUpdate true

# Create .git/rr-cache if it doesn't exist
mkdir -p .git/rr-cache

# Save current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Check if rerere-cache-upload branch exists
if git ls-remote --heads origin rerere-cache-upload | grep -q rerere-cache-upload; then
    echo "Found rerere cache on GitHub, downloading..."

    # Fetch the branch
    git fetch origin rerere-cache-upload

    # Count before
    LOCAL_BEFORE=$(find .git/rr-cache -type d -mindepth 1 2>/dev/null | wc -l)

    # Use git worktree to checkout the cache branch without affecting current branch
    TEMP_WORKTREE=$(mktemp -d)
    git worktree add "$TEMP_WORKTREE" origin/rerere-cache-upload 2>/dev/null || {
        # Fallback: checkout files directly
        git checkout origin/rerere-cache-upload -- rr-cache 2>/dev/null || true
        if [ -d rr-cache ]; then
            cp -r rr-cache/* .git/rr-cache/ 2>/dev/null || true
            rm -rf rr-cache
        fi
        LOCAL_AFTER=$(find .git/rr-cache -type d -mindepth 1 2>/dev/null | wc -l)
        NEW_RESOLUTIONS=$((LOCAL_AFTER - LOCAL_BEFORE))
        echo "✅ Downloaded rerere cache"
        echo "   New resolutions added: $NEW_RESOLUTIONS"
        echo "   Total resolutions: $LOCAL_AFTER"
        exit 0
    }

    # Copy cache from worktree
    if [ -d "$TEMP_WORKTREE/rr-cache" ]; then
        cp -r "$TEMP_WORKTREE/rr-cache"/* .git/rr-cache/ 2>/dev/null || true
    fi

    # Cleanup worktree
    git worktree remove "$TEMP_WORKTREE" --force 2>/dev/null || true

    # Count after
    LOCAL_AFTER=$(find .git/rr-cache -type d -mindepth 1 2>/dev/null | wc -l)
    NEW_RESOLUTIONS=$((LOCAL_AFTER - LOCAL_BEFORE))

    echo "✅ Downloaded rerere cache"
    echo "   New resolutions added: $NEW_RESOLUTIONS"
    echo "   Total resolutions: $LOCAL_AFTER"
else
    echo "No rerere cache found on GitHub (this is normal for first run)"
    echo "After resolving conflicts, use upload-rerere-cache.sh to share your resolutions"
fi
