#!/usr/bin/env bash
# 通过 curl 向 vLLM 的 OpenAI 兼容端点 /v1/chat/completions 发送请求。
#
# 用法:
#   ./send_curl_openai.sh                    # 使用默认 json 和默认地址
#   ./send_curl_openai.sh openai_request.json
#   BASE_URL=http://localhost:8080 ./send_curl_openai.sh
#
# json 里若 "stream": true,服务端返回 SSE 流;本脚本用 --no-buffer 让其实时打印。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_FILE="${1:-$SCRIPT_DIR/openai_request.json}"

# 服务地址与鉴权(可用环境变量覆盖)
BASE_URL="${BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-EMPTY}"

curl --no-buffer -sS -X POST "${BASE_URL}/v1/chat/completions" \
  -H "content-type: application/json" \
  -H "authorization: Bearer ${API_KEY}" \
  --data-binary "@${JSON_FILE}"
echo
