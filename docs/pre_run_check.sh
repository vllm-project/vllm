if [ "$READTHEDOCS_VERSION_TYPE" != "external" ]; then
  echo "Not a PR build (version type=$READTHEDOCS_VERSION_TYPE); skipping pre-run-check gate."
  exit 0
fi

echo "Checking for changes to docs-affecting files vs origin/main..."
DOCS_PATHS=(
  docs/                       # Actual docs content
  examples/                   # Examples are rendered in docs
  vllm/                       # API & CLI reference
  requirements/test/cuda.txt  # CLI reference (see docs/mkdocs/hooks/generate_argparse.py)
  mkdocs.yaml                 # Affects build process
  .readthedocs.yaml           # Affects build process
  requirements/docs.txt       # Affects build process
  requirements/docs.in        # Affects build process
)
if git diff --quiet origin/main -- "${DOCS_PATHS[@]}"; then
  echo "No docs-affecting files changed vs origin/main; cancelling build."
  # See https://docs.readthedocs.com/platform/latest/guides/build/skip-build.html for info on exit code
  exit 183
fi
echo "Docs-affecting files changed; continuing pre-run-check."
echo "Checking pre-commit/pre-run-check status..."
MAX_WAIT=300
INTERVAL=60
ELAPSED=0
# Use a GitHub token if provided to raise the API rate limit (60 -> 5000
# requests/hour). Set GITHUB_TOKEN in the Read the Docs environment variables.
CURL_AUTH=()
if [ -n "$GITHUB_TOKEN" ]; then
  CURL_AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN")
fi
while :; do
  RAW=$(curl -sS "${CURL_AUTH[@]}" -w "\n%{http_code}" "https://api.github.com/repos/vllm-project/vllm/commits/${READTHEDOCS_GIT_COMMIT_HASH}/check-runs?check_name=pre-run-check&filter=latest")
  HTTP_CODE=$(printf %s "$RAW" | tail -n1)
  BODY=$(printf %s "$RAW" | sed '$d')
  if [ "$HTTP_CODE" != "200" ]; then
    echo "GitHub API returned HTTP $HTTP_CODE (likely rate-limited); skipping pre-commit/pre-run-check gate."
    break
  fi
  STATUS=$(printf %s "$BODY" | python3 -c "import sys, json; r=json.load(sys.stdin).get(\"check_runs\",[]); print((r[0].get(\"status\") or \"\") if r else \"none\")")
  CONCLUSION=$(printf %s "$BODY" | python3 -c "import sys, json; r=json.load(sys.stdin).get(\"check_runs\",[]); print((r[0].get(\"conclusion\") or \"\") if r else \"\")")
  CHECK_URL=$(printf %s "$BODY" | python3 -c "import sys, json; r=json.load(sys.stdin).get(\"check_runs\",[]); print((r[0].get(\"html_url\") or \"\") if r else \"\")")
  if [ "$STATUS" = "none" ]; then
    echo "no pre-commit/pre-run-check found for this commit; skipping gate."
    break
  fi
  if [ -n "$CONCLUSION" ]; then
    echo "pre-commit/pre-run-check conclusion: $CONCLUSION"
    if [ "$CONCLUSION" = "failure" ] || [ "$CONCLUSION" = "cancelled" ] || [ "$CONCLUSION" = "timed_out" ]; then
      echo "pre-commit/pre-run-check did not pass; skipping docs build."
      if [ -n "$CHECK_URL" ]; then
        echo "pre-commit/pre-run-check failure reason: $CHECK_URL"
      fi
      exit 1
    fi
    break
  fi
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "pre-commit/pre-run-check status=$STATUS after ${MAX_WAIT}s; skipping gate."
    break
  fi
  echo "pre-commit/pre-run-check status=$STATUS; waiting ${INTERVAL}s..."
  sleep "$INTERVAL"
  ELAPSED=$((ELAPSED + INTERVAL))
done
