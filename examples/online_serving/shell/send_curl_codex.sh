#!/usr/bin/env bash
# 通过 curl 向 vLLM 的 Responses 端点 /v1/responses 发送 Codex 请求。
#
# 请求体来自 codex 的 jsonl 轨迹最后一行(payload.request.body),已抽取为
# codex_request.json。参考 examples/online_serving/shell/send_curl_anthropic.sh。
#
# 用法:
#   ./send_curl_codex.sh                  # 使用默认 json 和默认地址
#   ./send_curl_codex.sh my_request.json  # 指定请求体 json
#   BASE_URL=http://localhost:10000 ./send_curl_codex.sh
#
# json 里若 "stream": true，服务端会返回 SSE 流；本脚本用 --no-buffer 让其实时打印。

set -euo pipefail

# 请求体 JSON 文件（默认取脚本同目录下的 codex_request.json）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_FILE="${1:-$SCRIPT_DIR/codex_request.json}"

# 服务地址与鉴权（可用环境变量覆盖）
BASE_URL="${BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-EMPTY}"

curl --no-buffer -sS -X POST "${BASE_URL}/v1/responses" \
  -H "content-type: application/json" \
  -H "authorization: Bearer ${API_KEY}" \
  --data-binary "@${JSON_FILE}"
echo
