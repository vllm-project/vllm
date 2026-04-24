---
name: check-github-actions-status
description: Check GitHub Actions workflow run status, view logs, and identify errors. Useful for debugging failed workflows, monitoring CI/CD pipelines, or checking build/test/perf results. Use when the user asks about workflow status, wants to see logs, or needs to debug failed runs.
---

# Check GitHub Actions Workflow Status

Monitor workflow runs, view logs, and identify errors.

## Prerequisites

Check dependencies before proceeding:

```bash
# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
  echo "GitHub CLI (gh) is not installed."
  echo ""
  echo "Install it with:"
  echo "  Ubuntu/Debian: sudo apt-get install gh"
  echo "  macOS: brew install gh"
  echo "  Or visit: https://cli.github.com/"
  exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
  echo "GitHub CLI is not authenticated."
  echo "Run: gh auth login"
  exit 1
fi
```

## Common Operations

Always scope commands to this repo:

```bash
REPO="cohere-ai/vllm-cohere"
```

### List Recent Workflow Runs

```bash
# List recent runs for all workflows
gh run list -R "$REPO"

# List runs for specific workflow
gh run list -R "$REPO" --workflow=build-and-push.yaml
gh run list -R "$REPO" --workflow=build-and-test.yaml
gh run list -R "$REPO" --workflow=build-and-eval.yaml
gh run list -R "$REPO" --workflow=build-and-bench.yaml

# List runs for current branch
gh run list -R "$REPO" --branch $(git branch --show-current)

# Limit number of results
gh run list -R "$REPO" --limit 10
```

### Check Specific Run Status

```bash
# Get run ID from list, then check status
gh run view <RUN_ID> -R "$REPO"

# View in browser
gh run view <RUN_ID> -R "$REPO" --web

# Watch status (updates in real-time)
gh run watch <RUN_ID> -R "$REPO"
```

### View Workflow Logs

```bash
# View logs for a specific run
gh run view <RUN_ID> -R "$REPO" --log

# View logs for failed jobs only
gh run view <RUN_ID> -R "$REPO" --log-failed

# View logs for a specific job
gh run view <RUN_ID> -R "$REPO" --job <JOB_ID> --log
```

### Find Failed Runs

```bash
# List only failed runs
gh run list -R "$REPO" --status failure

# List failed runs for specific workflow
gh run list -R "$REPO" --workflow=build-and-test.yaml --status failure
gh run list -R "$REPO" --workflow=build-and-eval.yaml --status failure
gh run list -R "$REPO" --workflow=build-and-bench.yaml --status failure
```

### View Job Details

```bash
# View all jobs in a run
gh run view <RUN_ID> -R "$REPO"

# View specific job details
gh run view <RUN_ID> -R "$REPO" --job <JOB_ID>
```

## Workflow-Specific Checks

### build-and-push Status

```bash
# Check latest build-and-push run
gh run list -R "$REPO" --workflow=build-and-push.yaml --limit 1

# View logs for latest run
LATEST_RUN=$(gh run list -R "$REPO" --workflow=build-and-push.yaml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $LATEST_RUN -R "$REPO" --log
```

**Common issues to check**:
- Docker build failures
- Image push errors
- Cache issues
- Authentication problems

### build-and-test Status

```bash
# Check latest build-and-test run
gh run list -R "$REPO" --workflow=build-and-test.yaml --limit 1

# View failed tests
LATEST_RUN=$(gh run list -R "$REPO" --workflow=build-and-test.yaml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $LATEST_RUN -R "$REPO" --log-failed
```

**Common issues to check**:
- Test failures
- GPU availability
- Model download errors
- Timeout issues

### build-and-eval Status

```bash
# Check latest build-and-eval run
gh run list -R "$REPO" --workflow=build-and-eval.yaml --limit 1

# View failed jobs
LATEST_RUN=$(gh run list -R "$REPO" --workflow=build-and-eval.yaml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $LATEST_RUN -R "$REPO" --log-failed
```

**Common issues to check**:
- Eval harness failures (`lm_eval`, `bee_eval`)
- GPU availability
- Model download errors
- Timeout issues

### build-and-bench Status

```bash
# Check latest build-and-bench run
gh run list -R "$REPO" --workflow=build-and-bench.yaml --limit 1

# View failed jobs
LATEST_RUN=$(gh run list -R "$REPO" --workflow=build-and-bench.yaml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run view $LATEST_RUN -R "$REPO" --log-failed
```

**Common issues to check**:
- Perf job scheduling / runner availability
- Model download or tokenizer setup errors
- Benchmark script failures
- Result upload / `ci_dump` write issues

## Error Analysis Workflow

When a workflow fails:

1. **Get run details**:
   ```bash
   gh run view <RUN_ID> -R "$REPO"
   ```

2. **Identify failed jobs**:
   ```bash
   gh run view <RUN_ID> -R "$REPO" --json jobs --jq '.[] | select(.conclusion=="failure") | {name: .name, id: .databaseId}'
   ```

3. **View failed job logs**:
   ```bash
   gh run view <RUN_ID> -R "$REPO" --log-failed
   ```

4. **Extract error patterns**:
   - Look for "Error:", "Failed:", "Exception:" patterns
   - Check for timeout messages
   - Identify missing dependencies or permissions

## Useful Queries

### Check Current Branch Runs

```bash
BRANCH=$(git branch --show-current)
gh run list -R "$REPO" --branch "$BRANCH" --limit 5
```

### Monitor Running Workflows

```bash
# List currently running workflows
gh run list -R "$REPO" --status in_progress

# Watch a specific run
gh run watch <RUN_ID> -R "$REPO"
```

### Compare Runs

```bash
# View two runs side by side
gh run view <RUN_ID_1> -R "$REPO" --web &
gh run view <RUN_ID_2> -R "$REPO" --web &
```

## Output Formatting

For better readability, use JSON queries:

```bash
# Get run summary
gh run view <RUN_ID> -R "$REPO" --json status,conclusion,workflowName,headBranch,startedAt,completedAt

# Get failed job names
gh run view <RUN_ID> -R "$REPO" --json jobs --jq '.[] | select(.conclusion=="failure") | .name'

# Get run duration
gh run view <RUN_ID> -R "$REPO" --json startedAt,completedAt --jq '(.completedAt | fromdateiso8601) - (.startedAt | fromdateiso8601)'
```

## Integration Tips

- After triggering a workflow (using `trigger-github-actions` skill), offer to check status
- When user reports a failure, use this skill to investigate
- For debugging, focus on failed job logs first
- Use `--web` flag to open detailed view in browser when needed
