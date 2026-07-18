#!/usr/bin/env bash
# 通过 curl 向 vLLM 的 Anthropic 兼容端点 /v1/messages 发送请求。
#
# 用法:
#   ./send_curl.sh                  # 使用默认 json 和默认地址
#   ./send_curl.sh yxing_test.json  # 指定请求体 json
#   BASE_URL=http://localhost:10000 ./send_curl.sh
#
# json 里若 "stream": true，服务端会返回 SSE 流；本脚本用 --no-buffer 让其实时打印。

set -euo pipefail

# 请求体 JSON 文件（默认取脚本同目录下的 yxing_test.json）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_FILE="${1:-$SCRIPT_DIR/claude_code_request.json}"

# 服务地址与鉴权（可用环境变量覆盖）
BASE_URL="${BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-EMPTY}"

curl --no-buffer -sS -X POST "${BASE_URL}/v1/messages" \
  -H "content-type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -H "anthropic-version: 2023-06-01" \
  --data-binary "@${JSON_FILE}"
echo
