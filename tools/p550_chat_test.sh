#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

host="${HOST:-127.0.0.1}"
port="${PORT:-8000}"
model="${MODEL:-p550-tiny}"
prompt="${*:-Hello from P550. Give a short reply.}"
base_url="http://${host}:${port}"
url="${base_url}/v1/chat/completions"
wait_seconds="${WAIT_SECONDS:-240}"
request_timeout="${REQUEST_TIMEOUT:-300}"

printf 'Waiting for vLLM service at %s/health' "$base_url" >&2
for i in $(seq 1 "$wait_seconds"); do
    if python3 - "$base_url/health" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
urllib.request.urlopen(sys.argv[1], timeout=2).read()
PY
    then
        printf '\nService is ready after %s seconds.\n' "$i" >&2
        break
    fi
    if [ "$i" -eq "$wait_seconds" ]; then
        printf '\nService did not become ready within %s seconds.\n' "$wait_seconds" >&2
        exit 1
    fi
    printf '.' >&2
    sleep 1
done

python3 - "$url" "$model" "$prompt" "$request_timeout" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url, model, prompt, timeout_s = sys.argv[1:5]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0.0,
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=int(timeout_s)) as response:
        data = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    print(exc.read().decode("utf-8"), file=sys.stderr)
    raise SystemExit(exc.code) from exc
except urllib.error.URLError as exc:
    print(f"Failed to connect to {url}: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

print(json.dumps(data, indent=2, ensure_ascii=False))
try:
    print("\nassistant:", data["choices"][0]["message"]["content"])
except Exception:
    pass
PY
