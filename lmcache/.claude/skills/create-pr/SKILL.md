---
name: create-pr
description: Create a GitHub pull request from the current branch to the upstream repo, using the repo's PR template
allowed-tools: Bash, Read, Glob, Grep, Agent, mcp__github__create_pull_request
argument-hint: "[base_branch] [--draft] [--title 'PR title']"
---

# Create Pull Request

Create a GitHub pull request from the current branch to the upstream repository.

## Arguments

`$ARGUMENTS` may contain:
- First positional arg: base branch to merge into (default: auto-detect from upstream's default branch)
- `--draft`: create as a draft PR
- `--title 'PR title'`: override the PR title (default: auto-generated from changes)

## Steps

### 1. Gather context

Run these in parallel:

```bash
# Check remotes — identify the fork remote and the upstream remote
git remote -v

# Current branch and tracking info
git branch -vv --contains HEAD

# Untracked/unstaged changes (warn user if dirty)
git status -s

# Default branch of the upstream repo
gh api repos/<UPSTREAM_OWNER>/<UPSTREAM_REPO> --jq '.default_branch'
```

Determine:
- **Fork remote**: the remote pointing to the user's fork (usually `local` or `origin` if the user's fork)
- **Upstream remote**: the remote pointing to the upstream repo (look for the canonical org/repo)
- **Fork owner**: extracted from the fork remote URL (e.g., `alice` from `git@github.com:alice/LMCache.git`)
- **Upstream owner/repo**: extracted from the upstream remote URL
- **Base branch**: from `$ARGUMENTS` or the upstream repo's default branch

### 2. Check the branch is pushed

Verify the current branch is pushed to the fork remote. If not, warn the user and stop — do NOT push without explicit permission.

### 3. Find and read the PR template

```
Glob pattern: .github/PULL_REQUEST_TEMPLATE.md
Glob pattern: .github/PULL_REQUEST_TEMPLATE/**/*.md
Glob pattern: .github/pull_request_template.md
```

Read the template to understand required sections.

### 4. Analyze changes

```bash
# Commits on this branch vs the base
git log --oneline <upstream_remote>/<base_branch>..HEAD

# Diff stat for summary
git diff <upstream_remote>/<base_branch>..HEAD --stat
```

If there are many commits, also read the full diff to understand the changes:
```bash
git diff <upstream_remote>/<base_branch>..HEAD
```

### 5. Draft the PR

- **Title**: Use `--title` from arguments if provided. Otherwise, generate a concise title (under 70 chars) summarizing the change.
- **Body**: Fill in the PR template sections based on the changes. Focus on:
  - **Why** the change is needed (motivation, problem being solved)
  - **User-facing changes** (new CLI args, changed behavior, breaking changes)
  - Do NOT enumerate per-file changes — reviewers can read the diff
  - Keep it concise and informative

Show the drafted title and body to the user and ask for confirmation before creating the PR.

### 6. Create the PR

Use the `mcp__github__create_pull_request` tool:
- `owner`: upstream repo owner
- `repo`: upstream repo name
- `head`: `<fork_owner>:<branch_name>`
- `base`: the base branch
- `title`: the PR title
- `body`: the PR body
- `draft`: true if `--draft` was specified

### 7. Report

Print the PR URL so the user can review it.

## Important Notes

- Never push code or create commits — this skill only creates the PR from already-pushed branches.
- Always show the PR title and body to the user for confirmation before creating.
- If the working tree is dirty, warn the user that uncommitted changes won't be included.
- Respect the PR template structure — fill in all sections, use checkboxes where the template has them.
