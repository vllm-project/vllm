#!/bin/bash
# Cleanup containers launched for multiprocessing tests
# This script should always be called, even on failure

LMCACHE_CONTAINER_NAME="${LMCACHE_CONTAINER_NAME:-lmcache-mp-test}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-mp-test}"
VLLM_BASELINE_CONTAINER_NAME="${VLLM_BASELINE_CONTAINER_NAME:-vllm-baseline-test}"

echo "=== Cleaning up containers ==="

# Function to stop and remove a container
cleanup_container() {
    local container_name="$1"
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "Stopping and removing container: $container_name"
        
        # Try to get logs before cleanup (for debugging)
        echo "--- Container logs for $container_name ---"
        docker logs --tail 50 "$container_name" 2>&1 || true
        echo "--- End of logs ---"
        
        # Stop container (with timeout)
        docker stop --timeout 10 "$container_name" 2>/dev/null || true
        
        # Remove container
        docker rm -f "$container_name" 2>/dev/null || true
        
        echo "Container $container_name cleaned up"
    else
        echo "Container $container_name not found (already cleaned up or never started)"
    fi
}

# Cleanup all containers
cleanup_container "$VLLM_BASELINE_CONTAINER_NAME"
cleanup_container "$VLLM_CONTAINER_NAME"
cleanup_container "$LMCACHE_CONTAINER_NAME"

echo "=== Cleanup complete ==="

# Wait for GPU memory to be fully released
echo "Waiting 5 seconds for GPU memory to be released..."
sleep 5

# List any remaining test containers (for debugging)
remaining=$(docker ps -a --format '{{.Names}}' | grep -E "(lmcache|vllm).*-(mp|baseline)-test" || true)
if [ -n "$remaining" ]; then
    echo "Warning: Some test containers may still exist:"
    echo "$remaining"
fi

