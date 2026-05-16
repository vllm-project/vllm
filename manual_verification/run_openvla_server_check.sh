#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

workspace_root="${WORKSPACE_ROOT:-/workspace}"
run_dir="${OPENVLA_RUN_DIR:-${workspace_root}/openvla_check}"
log_dir="${workspace_root}/logs"
script_dir="${repo_root}/manual_verification"
log="${log_dir}/openvla_server_check_$(date +%Y%m%d_%H%M%S).log"
server_log="${log_dir}/openvla_server_$(date +%Y%m%d_%H%M%S).log"
chat_template="${run_dir}/openvla_chat_template.jinja"

mkdir -p "${log_dir}" "${run_dir}"
export OPENVLA_RUN_DIR="${run_dir}"

{
  echo "OpenVLA server check"
  echo "repo_root=${repo_root}"
  echo "run_dir=${run_dir}"
  echo "commit=$(git rev-parse --short HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "start=$(date -Is)"
  echo

  if [ -f "${workspace_root}/.hf_env" ]; then
    . "${workspace_root}/.hf_env"
  fi
  export VLLM_ALLOW_INSECURE_SERIALIZATION=1

  if [ ! -f "${run_dir}/cases.json" ] || [ ! -f "${run_dir}/hf_artifacts.pt" ]; then
    echo "Missing ${run_dir}/cases.json or ${run_dir}/hf_artifacts.pt."
    echo "Run: bash manual_verification/run_openvla_check.sh"
    exit 1
  fi

  cat >"${chat_template}" <<'EOF'
{% for message in messages %}{% if message.role == "user" %}{% if message.content is string %}<s>In: {{ message.content }}
Out:{% else %}{% for content in message.content %}{% if content.type == "text" %}<s>In: {{ content.text }}
Out:{% endif %}{% endfor %}{% endif %}{% endif %}{% endfor %}
EOF

  .venv/bin/vllm serve openvla/openvla-7b \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 512 \
    --enforce-eager \
    --gpu-memory-utilization 0.75 \
    --chat-template "${chat_template}" \
    --port 8080 >"${server_log}" 2>&1 &
  server_pid=$!

  cleanup() {
    if kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  echo "server_log=${server_log}"
  echo "waiting for server"
  for _ in $(seq 1 120); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "server exited before becoming ready"
      tail -n 200 "${server_log}" || true
      exit 1
    fi
    if .venv/bin/python - <<'PY' >/dev/null 2>&1
import urllib.request

urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
PY
    then
      echo "server ready"
      break
    fi
    sleep 5
  done

  if ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import urllib.request

urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
PY
  then
    echo "server did not become ready"
    tail -n 200 "${server_log}" || true
    exit 1
  fi

  .venv/bin/python "${script_dir}/run_server_openvla.py"

  echo "server_result=${run_dir}/server_result.json"
  echo "end=$(date -Is)"
} 2>&1 | tee "${log}"

echo "LOG=${log}"
