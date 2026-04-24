#!/bin/bash
# Upload local rerere cache to GitHub for use in Actions

set -e

if [ ! -d .git/rr-cache ]; then
    echo "❌ No rerere cache found locally (.git/rr-cache doesn't exist)"
    echo "Have you resolved any conflicts yet?"
    exit 1
fi

CACHE_COUNT=$(find .git/rr-cache -type d -mindepth 1 | wc -l)
echo "Found $CACHE_COUNT rerere resolutions in local cache"

if [ "$CACHE_COUNT" -eq 0 ]; then
    echo "❌ Rerere cache is empty"
    exit 1
fi

echo "Creating branch with rerere cache..."

# Create orphan branch for cache storage
git checkout --orphan rerere-cache-upload
git rm -rf . 2>/dev/null || true

# Copy rerere cache
cp -r .git/rr-cache ./rr-cache

# Commit
git add rr-cache/
git commit -m "Update rerere cache - $(date -u +%Y-%m-%dT%H:%M:%SZ)

Resolutions: $CACHE_COUNT
"

# Push
echo "Pushing rerere cache to GitHub..."
git push -f origin rerere-cache-upload

# Go back to previous branch
git checkout -

echo "✅ Rerere cache uploaded to branch: rerere-cache-upload"
echo ""
echo "GitHub Actions will now use these resolutions in future rebases"
