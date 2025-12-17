# Git Bisect with Buildkite

Automated git bisect using Buildkite CI builds to find the first bad commit.

## Architecture

This system consists of two pipelines that work together:

### 1. Driver Pipeline (`bisect-driver-pipeline.yaml`)
Orchestrates the bisect process by:
- Querying the bad build to identify failed jobs
- Running git bisect locally
- Triggering validation builds at each bisect step
- Reporting the first bad commit

### 2. Validator Pipeline (`bisect-validator-pipeline.yaml`)
Runs only specific test jobs:
- Accepts `TARGET_JOBS` environment variable (comma-separated job labels)
- Dynamically generates pipeline with only requested tests
- Much faster than running full test suite

## Setup

### 1. Create Buildkite Pipelines

Create two pipelines in the Buildkite UI:

**Validation Pipeline:**
- Name: `vLLM Validation`
- Slug: `validation`
- Pipeline file: `.buildkite/bisect/bisect-validator-pipeline.yaml`
- Description: "Runs specific test jobs for bisect validation"

**Driver Pipeline:**
- Name: `vLLM Bisect Driver`
- Slug: `bisect-driver`
- Pipeline file: `.buildkite/bisect/bisect-driver-pipeline.yaml`
- Description: "Automated git bisect using Buildkite builds"

### 2. Set Up API Token

The driver needs a Buildkite API token with `read_builds` and `write_builds` scopes:

1. Go to https://buildkite.com/user/api-access-tokens
2. Create a new token with scopes: `read_builds`, `write_builds`
3. Store the token securely (e.g., in CI environment or secrets manager)

## Usage

### Option 1: Local Execution

Run the bisect driver locally:

```bash
# Set up API token
export BUILDKITE_TOKEN=your_token_here

# Basic bisect (runs all tests at each step)
python .buildkite/bisect/bisect_driver.py <good_sha> <bad_sha>

# Smart bisect (runs only failed jobs)
python .buildkite/bisect/bisect_driver.py <good_sha> <bad_sha> \
  --validation-pipeline validation

# Specify exact build to get failed jobs from
python .buildkite/bisect/bisect_driver.py <good_sha> <bad_sha> \
  --bad-build 12345 \
  --validation-pipeline validation
```

### Option 2: Run on CI

Trigger the driver pipeline via Buildkite UI or API:

**Via Buildkite UI:**
1. Go to the "vLLM Bisect Driver" pipeline
2. Click "New Build"
3. Set environment variables:
   - `GOOD_SHA`: Known good commit
   - `BAD_SHA`: Known bad commit
   - `BAD_BUILD`: (Optional) Build number with failures
4. Trigger the build

**Via API:**
```bash
curl -X POST \
  "https://api.buildkite.com/v2/organizations/vllm/pipelines/bisect-driver/builds" \
  -H "Authorization: Bearer $BUILDKITE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "commit": "HEAD",
    "branch": "main",
    "env": {
      "GOOD_SHA": "abc123def",
      "BAD_SHA": "def456abc",
      "BAD_BUILD": "12345"
    }
  }'
```

## Files

```
.buildkite/bisect/
├── README.md                       # This file
├── bisect_driver.py                # Bisect driver script
├── bisect-driver-pipeline.yaml     # Pipeline to run driver on CI
├── bisect_validator.py             # Generates filtered test steps
└── bisect-validator-pipeline.yaml  # Pipeline that runs specific tests
```

## How It Works

1. **Query Failed Jobs:**
   - Driver queries the bad build (or finds build from bad SHA)
   - Extracts labels of all failed test jobs

2. **Start Git Bisect:**
   - Initializes `git bisect` with good and bad commits
   - Git selects a commit in the middle

3. **Test Each Commit:**
   - Driver triggers validation pipeline for the current commit
   - Passes `TARGET_JOBS` env var with failed job labels
   - Validation pipeline generates and runs only those tests
   - Driver waits for build to complete

4. **Mark Good/Bad:**
   - If tests pass → `git bisect good`
   - If tests fail → `git bisect bad`
   - Git selects next commit

5. **Repeat:**
   - Process repeats until git identifies the first bad commit

6. **Report:**
   - Driver outputs the first bad commit SHA
   - Resets bisect state

## Examples

### Example 1: Simple Bisect

You know commit `abc123` is good and `def456` is bad:

```bash
python .buildkite/bisect/bisect_driver.py abc123 def456 --validation-pipeline validation
```

### Example 2: Using Build Number

You have a failing build #12345 and know the previous release was good:

```bash
python .buildkite/bisect/bisect_driver.py v0.6.0 HEAD \
  --bad-build 12345 \
  --validation-pipeline validation
```

### Example 3: Full Test Suite

Run complete test suite at each step (slower but comprehensive):

```bash
python .buildkite/bisect/bisect_driver.py abc123 def456 --pipeline test-pipeline
```

## Troubleshooting

**Problem:** No failed jobs found in bad build
- Solution: Verify the build number is correct and has failed jobs
- Check that you're using the right pipeline slug

**Problem:** Bisect interrupted
- Solution: Resume with: `python .buildkite/bisect/bisect_driver.py --resume`
- Or reset with: `git bisect reset`

**Problem:** Test job names don't match
- Solution: Job labels must match exactly (case-insensitive)
- Check available labels with: `yq '.steps[].label' .buildkite/test-pipeline.yaml`

**Problem:** API token not set
- Solution: `export BUILDKITE_TOKEN=your_token_here`

## Advanced Options

Run `python .buildkite/bisect/driver.py --help` for all options:

- `--org`: Buildkite organization slug (default: vllm)
- `--pipeline`: Main pipeline slug (default: ci)
- `--validation-pipeline`: Validation pipeline slug
- `--bad-build`: Build number to query for failures
- `--branch`: Git branch (default: main)
- `--poll-interval`: Seconds between build polls (default: 30)
- `--timeout`: Build timeout in minutes (default: 120)
- `--resume`: Resume interrupted bisect
