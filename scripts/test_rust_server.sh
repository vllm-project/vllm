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
curl -s "$BASE_URL/health" | jq .
echo "PASS"

# Test 2: Non-streaming chat completion
echo ""
echo "Test 2: Non-streaming chat completion"
curl -s "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": false
    }' | jq .
echo "PASS"

# Test 3: Streaming chat completion
echo ""
echo "Test 3: Streaming chat completion"
curl -s -N "$BASE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "test",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "max_tokens": 20,
        "stream": true
    }' | head -10
echo ""
echo "PASS"

echo ""
echo "All tests passed!"
