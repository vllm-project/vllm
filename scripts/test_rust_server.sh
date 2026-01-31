#!/bin/bash
set -e

# Quick integration test for Rust API server
# Requires: running gRPC server and Rust API server

HTTP_PORT="${HTTP_PORT:-8000}"
BASE_URL="http://localhost:$HTTP_PORT"

echo "Testing Rust API Server at $BASE_URL"

# Test 1: Health check
echo ""
echo "Test 1: Health check"
RESPONSE=$(curl -sf "$BASE_URL/health") || { echo "FAIL: Health check failed"; exit 1; }
echo "$RESPONSE" | jq .
if echo "$RESPONSE" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL: Health check response missing status:healthy"
    exit 1
fi

# Test 2: Non-streaming chat completion
echo ""
echo "Test 2: Non-streaming chat completion"
RESPONSE=$(curl -sf "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": false
    }') || { echo "FAIL: Chat completion request failed"; exit 1; }
echo "$RESPONSE" | jq .
if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL: Response missing choices[0].message.content"
    exit 1
fi

# Test 3: Streaming chat completion
echo ""
echo "Test 3: Streaming chat completion"
RESPONSE=$(curl -sf -N "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "max_tokens": 20,
        "stream": true
    }') || { echo "FAIL: Streaming request failed"; exit 1; }
echo "$RESPONSE" | head -10
# Check that we got SSE data lines
if echo "$RESPONSE" | grep -q "^data:"; then
    echo "PASS"
else
    echo "FAIL: Response missing SSE data lines"
    exit 1
fi

echo ""
echo "All tests passed!"
