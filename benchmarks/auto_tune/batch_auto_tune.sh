#!/bin/bash

INPUT_JSON="$1"
GCS_PATH="$2" # Optional GCS path for uploading results for each run

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
AUTOTUNE_SCRIPT="$SCRIPT_DIR/auto_tune.sh"

if [[ -z "$INPUT_JSON" ]]; then
  echo "Error: Input JSON file not provided."
  echo "Usage: $0 <path_to_json_file> [gcs_upload_path]"
  exit 1
fi

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "Error: File not found at '$INPUT_JSON'"
  exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' command not found. Please install jq to process the JSON input."
    exit 1
fi

if [[ -n "$GCS_PATH" ]] && ! command -v gcloud &> /dev/null; then
    echo "Error: 'gcloud' command not found, but a GCS_PATH was provided."
    exit 1
fi

SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_RUNS=()
SCRIPT_START_TIME=$(date +%s)

json_content=$(cat "$INPUT_JSON")
if ! num_runs=$(echo "$json_content" | jq 'length'); then
  echo "Error: Invalid JSON in $INPUT_JSON. 'jq' failed to get array length." >&2
  exit 1
fi

echo "Found $num_runs benchmark configurations in $INPUT_JSON."
echo "Starting benchmark runs..."
echo "--------------------------------------------------"

for i in $(seq 0 $(($num_runs - 1))); do
  run_object=$(echo "$json_content" | jq ".[$i]")

  RUN_START_TIME=$(date +%s)
  ENV_VARS_ARRAY=()
  # Dynamically create env vars from the JSON object's keys
  for key in $(echo "$run_object" | jq -r 'keys_unsorted[]'); do
    value=$(echo "$run_object" | jq -r ".$key")
    var_name=$(echo "$key" | tr '[:lower:]' '[:upper:]' | tr -cd 'A-Z0-9_')
    ENV_VARS_ARRAY+=("${var_name}=${value}")
  done

  echo "Executing run #$((i+1))/$num_runs with parameters: ${ENV_VARS_ARRAY[*]}"

  # Execute auto_tune.sh and capture output
  RUN_OUTPUT_FILE=$(mktemp)
  if env "${ENV_VARS_ARRAY[@]}" bash "$AUTOTUNE_SCRIPT" > >(tee -a "$RUN_OUTPUT_FILE") 2>&1; then
    STATUS="SUCCESS"
    ((SUCCESS_COUNT++))
  else
    STATUS="FAILURE"
    ((FAILURE_COUNT++))
    FAILED_RUNS+=("Run #$((i+1)): $(echo $run_object | jq -c .)")
  fi

  RUN_OUTPUT=$(<"$RUN_OUTPUT_FILE")
  rm "$RUN_OUTPUT_FILE"

  # Parse results and optionally upload them to GCS
  RUN_ID=""
  RESULTS=""
  GCS_RESULTS_URL=""
  if [[ "$STATUS" == "SUCCESS" ]]; then
    RESULT_FILE_PATH=$(echo "$RUN_OUTPUT" | grep 'RESULT_FILE=' | tail -n 1 | cut -d'=' -f2 | tr -s '/' || true)

    if [[ -n "$RESULT_FILE_PATH" && -f "$RESULT_FILE_PATH" ]]; then
      RUN_ID=$(basename "$(dirname "$RESULT_FILE_PATH")")
      RESULT_DIR=$(dirname "$RESULT_FILE_PATH")
      RESULTS=$(cat "$RESULT_FILE_PATH")

      if [[ -n "$GCS_PATH" ]]; then
        GCS_RESULTS_URL="${GCS_PATH}/${RUN_ID}"
        echo "Uploading results to GCS..."
        if gcloud storage rsync --recursive "$RESULT_DIR/" "$GCS_RESULTS_URL"; then
          echo "GCS upload successful."
        else
          echo "Warning: GCS upload failed for RUN_ID $RUN_ID."
        fi
      fi
    else
      echo "Warning: Could not find result file for a successful run."
      STATUS="WARNING_NO_RESULT_FILE"
    fi
  fi

  # Add the results back into the JSON object for this run
  json_content=$(echo "$json_content" | jq --argjson i "$i" --arg run_id "$RUN_ID" --arg status "$STATUS" --arg results "$RESULTS" --arg gcs_results "$GCS_RESULTS_URL" \
    '.[$i] += {run_id: $run_id, status: $status, results: $results, gcs_results: $gcs_results}')

  RUN_END_TIME=$(date +%s)
  echo "Run finished in $((RUN_END_TIME - RUN_START_TIME)) seconds. Status: $STATUS"
  echo "--------------------------------------------------"

  # Save intermediate progress back to the file
  echo "$json_content" > "$INPUT_JSON.tmp" && mv "$INPUT_JSON.tmp" "$INPUT_JSON"

done

SCRIPT_END_TIME=$(date +%s)
echo "All benchmark runs completed in $((SCRIPT_END_TIME - SCRIPT_START_TIME)) seconds."
echo
echo "====================== SUMMARY ======================"
echo "Successful runs: $SUCCESS_COUNT"
echo "Failed runs:     $FAILURE_COUNT"
echo "==================================================="

if [[ $FAILURE_COUNT -gt 0 ]]; then
  echo "Details of failed runs (see JSON file for full parameters):"
  for failed in "${FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
fi

echo "Updated results have been saved to '$INPUT_JSON'."
