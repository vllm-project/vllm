#!/bin/bash

set -ex

# Clean up old nightly builds from DockerHub, keeping only the last 14 builds
# This script uses DockerHub API to list and delete old tags with "nightly-" prefix

# DockerHub API endpoint for vllm/vllm-openai repository
REPO_API_URL="https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags"

# Get DockerHub token from environment
if [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "Error: DOCKERHUB_TOKEN environment variable is not set"
    exit 1
fi

# Function to get all tags from DockerHub
get_all_tags() {
    local page=1
    local all_tags=""
    
    while true; do
        local response=$(curl -s -H "Authorization: Bearer $DOCKERHUB_TOKEN" \
            "$REPO_API_URL?page=$page&page_size=100")
        
        # Get both last_updated timestamp and tag name, separated by |
        local tags=$(echo "$response" | jq -r '.results[] | select(.name | startswith("nightly-")) | "\(.last_updated)|\(.name)"')
        
        if [ -z "$tags" ]; then
            break
        fi
        
        all_tags="$all_tags$tags"$'\n'
        page=$((page + 1))
    done
    
    # Sort by timestamp (newest first) and extract just the tag names
    echo "$all_tags" | sort -r | cut -d'|' -f2
}

delete_tag() {
    local tag_name="$1"
    echo "Deleting tag: $tag_name"
    
    local delete_url="https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/$tag_name"
    local response=$(curl -s -X DELETE -H "Authorization: Bearer $DOCKERHUB_TOKEN" "$delete_url")
    
    if echo "$response" | jq -e '.detail' > /dev/null 2>&1; then
        echo "Warning: Failed to delete tag $tag_name: $(echo "$response" | jq -r '.detail')"
    else
        echo "Successfully deleted tag: $tag_name"
    fi
}

# Get all nightly- prefixed tags, sorted by last_updated timestamp (newest first)
echo "Fetching all tags from DockerHub..."
all_tags=$(get_all_tags)

if [ -z "$all_tags" ]; then
    echo "No tags found to clean up"
    exit 0
fi

# Count total tags
total_tags=$(echo "$all_tags" | wc -l)
echo "Found $total_tags tags"

# Keep only the last 14 builds (including the current one)
tags_to_keep=14
tags_to_delete=$((total_tags - tags_to_keep))

if [ $tags_to_delete -le 0 ]; then
    echo "No tags need to be deleted (only $total_tags tags found, keeping $tags_to_keep)"
    exit 0
fi

echo "Will delete $tags_to_delete old tags, keeping the newest $tags_to_keep"

# Get tags to delete (skip the first $tags_to_keep tags)
tags_to_delete_list=$(echo "$all_tags" | tail -n +$((tags_to_keep + 1)))

if [ -z "$tags_to_delete_list" ]; then
    echo "No tags to delete"
    exit 0
fi

# Delete old tags
echo "Deleting old tags..."
while IFS= read -r tag; do
    if [ -n "$tag" ]; then
        delete_tag "$tag"
        # Add a small delay to avoid rate limiting
        sleep 1
    fi
done <<< "$tags_to_delete_list"

echo "Cleanup completed successfully"
