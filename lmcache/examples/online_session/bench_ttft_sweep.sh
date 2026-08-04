#!/usr/bin/env bash
# Run the TTFT benchmark at several context sizes and collate the results.
# ---------------------------------------------------------------
BENCH="openai_chat_completion_client.py"   # path to your benchmark script
MASTER_OUT="all_ttft_results.jsonl"        # final merged log
CONTEXT_SIZES=(50 1000 2000 8000 16000 24000 32000 64000 96000 128000)

: > "$MASTER_OUT"               # truncate / create the final log file

for TOKENS in "${CONTEXT_SIZES[@]}"; do
  MAX_CTX=$((TOKENS))
  OUTFILE="ttft_${TOKENS}.jsonl"

  echo -e "\n▶︎ Running ${TOKENS}-token test (max_ctx_tokens=${MAX_CTX})"
  python "$BENCH" --max_ctx_tokens "$MAX_CTX" \
                  --num_following 1 \
                  --out "$OUTFILE" \
		  --model meta-llama/Llama-3.3-70B-Instruct \

  cat "$OUTFILE" >> "$MASTER_OUT"          # append to the master log
done

echo -e "\n✅  All done – combined results in $MASTER_OUT"

