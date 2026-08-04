#!/bin/bash
# Wait for vLLM servers to be ready
set -e

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-mp-test}"
VLLM_BASELINE_CONTAINER_NAME="${VLLM_BASELINE_CONTAINER_NAME:-vllm-baseline-test}"

# Function to wait for a single vLLM server
wait_for_vllm_server() {
    local port="$1"
    local container_name="$2"
    local description="$3"
    
    echo "=== Waiting for $description to be ready ==="
    echo "Port: $port"
    echo "Container: $container_name"
    echo "Max wait time: ${MAX_WAIT_SECONDS}s"
    
    start_time=$(date +%s)
    end_time=$((start_time + MAX_WAIT_SECONDS))
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ "$current_time" -ge "$end_time" ]; then
            echo "❌ Timeout: $description did not become ready within ${MAX_WAIT_SECONDS} seconds"
            echo ""
            echo "=== $description container logs ==="
            docker logs --tail 100 "$container_name" 2>&1 || true
            return 1
        fi
        
        # Check if container is still running
        if ! docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
            echo "❌ $description container is not running"
            echo ""
            echo "=== $description container logs ==="
            docker logs "$container_name" 2>&1 || true
            return 1
        fi
        
        # Check if vLLM health endpoint is responding
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "✅ $description is ready! (took ${elapsed}s)"
            return 0
        fi
        
        # Also try the models endpoint as a fallback
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "✅ $description is ready! (took ${elapsed}s)"
            return 0
        fi
        
        echo "Waiting for $description... (${elapsed}s elapsed)"
        sleep 5
    done
}

# Wait for vLLM with LMCache
if ! wait_for_vllm_server "$VLLM_PORT" "$VLLM_CONTAINER_NAME" "vLLM (with LMCache)"; then
    exit 1
fi

echo ""

# Wait for vLLM baseline (without LMCache)
if ! wait_for_vllm_server "$VLLM_BASELINE_PORT" "$VLLM_BASELINE_CONTAINER_NAME" "vLLM baseline (without LMCache)"; then
    exit 1
fi

echo ""
echo "=== All vLLM servers are ready ==="

