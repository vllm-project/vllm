#!/bin/bash
# Clean up old per-commit rocm-base tags from ECR Public, keeping a rolling
# window of the most recent N commits' tags plus the cache key tag.
#
# Usage: cleanup-ecr-rocm-base-tags.sh <ecr-image-ref> [window-size]
#   ecr-image-ref: full ECR reference of the base image (cache-key tag to preserve)
#   window-size:   number of recent commit tags to keep (default 300)
set -euo pipefail

ECR_IMAGE_REF="${1:?Usage: $0 <ecr-image-ref> [window-size]}"
WINDOW_SIZE="${2:-300}"
REPO_NAME="vllm-release-repo"
REGION="us-east-1"

# Extract the cache key tag (always preserved)
CACHE_TAG="${ECR_IMAGE_REF##*:}"

# Get image digest from the locally-pulled image
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$ECR_IMAGE_REF" | awk -F@ '{print $2}')
if [ -z "$DIGEST" ]; then
  echo "WARNING: Could not get digest for $ECR_IMAGE_REF, skipping cleanup"
  exit 0
fi

# Get all tags for this specific digest from ECR
IMAGE_DETAIL=$(aws ecr-public describe-images \
  --repository-name "$REPO_NAME" \
  --region "$REGION" \
  --image-ids imageDigest="$DIGEST" \
  --output json 2>/dev/null || echo '{"imageDetails":[]}')

# Extract all -rocm-base tags (excluding the cache key tag)
COMMIT_BASE_TAGS=$(echo "$IMAGE_DETAIL" | jq -r \
  --arg cache_tag "$CACHE_TAG" \
  '.imageDetails[0].imageTags[]? // empty
   | select(endswith("-rocm-base"))
   | select(. != $cache_tag)')

TAG_COUNT=$(echo "$COMMIT_BASE_TAGS" | grep -c . || true)
echo "Found $TAG_COUNT per-commit rocm-base tags (plus cache key tag: $CACHE_TAG)"

if [ "$TAG_COUNT" -le "$WINDOW_SIZE" ]; then
  echo "Within window ($WINDOW_SIZE), no cleanup needed"
  exit 0
fi

# Get the most recent N commit SHAs from git history
RECENT_COMMITS=$(git log --format=%H -n "$WINDOW_SIZE" 2>/dev/null | sort)
if [ -z "$RECENT_COMMITS" ]; then
  echo "WARNING: Could not get git history, skipping cleanup"
  exit 0
fi

# Identify tags to delete: commit SHA not in recent history
TAGS_TO_DELETE=""
KEEP_COUNT=0
DELETE_COUNT=0
while IFS= read -r tag; do
  [ -z "$tag" ] && continue
  COMMIT_SHA="${tag%-rocm-base}"
  if echo "$RECENT_COMMITS" | grep -q "^${COMMIT_SHA}$"; then
    KEEP_COUNT=$((KEEP_COUNT + 1))
  else
    TAGS_TO_DELETE="${TAGS_TO_DELETE}${tag}"$'\n'
    DELETE_COUNT=$((DELETE_COUNT + 1))
  fi
done <<< "$COMMIT_BASE_TAGS"

echo "Keeping $KEEP_COUNT tags (recent commits), deleting $DELETE_COUNT old tags"

if [ "$DELETE_COUNT" -eq 0 ]; then
  echo "Nothing to delete"
  exit 0
fi

# Delete in batches of 100 (ECR batch-delete-image limit)
echo "$TAGS_TO_DELETE" | grep -v '^$' | while mapfile -t -n 100 BATCH && [ ${#BATCH[@]} -gt 0 ]; do
  IMAGE_IDS=""
  for tag in "${BATCH[@]}"; do
    [ -z "$tag" ] && continue
    IMAGE_IDS="$IMAGE_IDS imageTag=$tag"
  done
  if [ -n "$IMAGE_IDS" ]; then
    aws ecr-public batch-delete-image \
      --repository-name "$REPO_NAME" \
      --region "$REGION" \
      --image-ids $IMAGE_IDS 2>/dev/null || echo "WARNING: batch-delete failed for some tags"
  fi
done

echo "Cleanup complete: deleted $DELETE_COUNT old rocm-base tags, kept $KEEP_COUNT + cache key"
