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
TAGS_TO_DELETE=()
KEEP_COUNT=0
while IFS= read -r tag; do
  [ -z "$tag" ] && continue
  COMMIT_SHA="${tag%-rocm-base}"
  if echo "$RECENT_COMMITS" | grep -q "^${COMMIT_SHA}$"; then
    KEEP_COUNT=$((KEEP_COUNT + 1))
  else
    TAGS_TO_DELETE+=("$tag")
  fi
done <<< "$COMMIT_BASE_TAGS"

DELETE_COUNT=${#TAGS_TO_DELETE[@]}

echo "Keeping $KEEP_COUNT tags (recent commits), deleting $DELETE_COUNT old tags"

if [ "$DELETE_COUNT" -eq 0 ]; then
  echo "Nothing to delete"
  exit 0
fi

# Delete in batches of 100 (ECR batch-delete-image limit)
TOTAL_DELETED=0
NUM_BATCHES=$(( (DELETE_COUNT + 99) / 100 ))
for ((i=0; i<DELETE_COUNT; i+=100)); do
  BATCH=("${TAGS_TO_DELETE[@]:i:100}")
  BATCH_SIZE=${#BATCH[@]}
  BATCH_NUM=$(( i/100 + 1 ))

  # Build JSON array for --image-ids
  IMAGE_IDS_JSON=$(printf '%s\n' "${BATCH[@]}" | jq -R '{imageTag: .}' | jq -s '.')

  echo "Deleting batch ${BATCH_NUM}/${NUM_BATCHES} ($BATCH_SIZE tags)..."

  RESULT=$(aws ecr-public batch-delete-image \
    --repository-name "$REPO_NAME" \
    --region "$REGION" \
    --image-ids "$IMAGE_IDS_JSON" \
    --output json 2>&1) || true

  # Parse response for success/failure counts
  DELETED=$(echo "$RESULT" | jq '.imageIds | length' 2>/dev/null || echo 0)
  FAILED=$(echo "$RESULT" | jq '.failures | length' 2>/dev/null || echo 0)

  if [ "$FAILED" -gt 0 ]; then
    FAILURE_REASONS=$(echo "$RESULT" | jq -r '.failures[0:3][] | "  \(.failureCode): \(.failureReason)"' 2>/dev/null || echo "  (could not parse response)")
    echo "WARNING: $FAILED deletions failed, $DELETED succeeded. Sample failures:"
    echo "$FAILURE_REASONS"
  else
    echo "  Deleted $DELETED tags"
  fi

  TOTAL_DELETED=$((TOTAL_DELETED + DELETED))
done

echo "Cleanup complete: deleted $TOTAL_DELETED/$DELETE_COUNT old rocm-base tags, kept $KEEP_COUNT + cache key"
