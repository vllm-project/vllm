#!/bin/bash

SCRIPT_DIR="/root/scripts"
INPUT_CSV="$SCRIPT_DIR/settings_vllm.csv"
VARS_FILE="$SCRIPT_DIR/server_vars.txt"
INPUT_SH="$SCRIPT_DIR/template_vllm_server.sh"
OUTPUT_SH="$SCRIPT_DIR/vllm_server.sh"
LOG_DIR="${LOG_DIR:-/root/logs}"
LOG_FILE="vllm_server.log"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$LOG_FILE"

## PRE-CHECKS
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HOME

python3 "$SCRIPT_DIR/generate_vars.py" "$INPUT_CSV"
if [[ $? -ne 0 ]]; then
    echo "Settings Error. Exiting!"
    exit -1
fi
## If vars file did not get generated, then we cannot proceed
if [ ! -f "$VARS_FILE" ]; then
    echo "Failure creating env. Exiting!"
    exit -1
fi
set -a
source "$VARS_FILE"

which envsubst &>/dev/null || envsubst() { eval "echo \"$(sed 's/"/\\"/g')\""; }

envsubst < "$INPUT_SH" > "$OUTPUT_SH"
sed -i "/#@VARS/ r $VARS_FILE" "$OUTPUT_SH"

echo "=== vLLM SERVER SCRIPT ==="
cat "$OUTPUT_SH"
echo "==========================="
chmod +x "$SCRIPT_DIR"/*.sh
touch "$LOG_FILE"
ln -sf /proc/self/fd/1 "$LOG_FILE"
ln -sf /dev/stdout "$LOG_FILE"
ln -sf /dev/stderr "$LOG_FILE"

if [ "$1" = "dummy" ]; then
    echo "Dummy run. Exiting!"
    exit 0
fi
"$OUTPUT_SH" > "$LOG_FILE" 2>&1