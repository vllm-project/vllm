#!/bin/bash
# Reusable rebase logic with rerere auto-resolution
# Returns exit codes:
#   0 = success (rebase completed)
#   1 = conflicts need manual resolution
#   2 = error

# Usage: rebase-with-rerere.sh <upstream_ref> <target_branch> [--github-actions]

UPSTREAM_REF="$1"
TARGET_BRANCH="$2"
GITHUB_ACTIONS_MODE="${3:-}"

if [ -z "$UPSTREAM_REF" ] || [ -z "$TARGET_BRANCH" ]; then
    echo "Usage: rebase-with-rerere.sh <upstream_ref> <target_branch> [--github-actions]"
    exit 2
fi

# Attempt rebase
if git rebase "$UPSTREAM_REF"; then
    echo "✅ Rebase completed successfully (no conflicts or all resolved by rerere)"

    if [ "$GITHUB_ACTIONS_MODE" = "--github-actions" ]; then
        echo "status=success" >> $GITHUB_OUTPUT
        echo "has_conflicts=false" >> $GITHUB_OUTPUT
    fi

    exit 0
else
    # Rebase failed - check if there are actual unresolved conflicts
    echo ""
    echo "Rebase paused - checking for conflicts..."
    echo ""

    CONFLICTED_FILES=$(git diff --name-only --diff-filter=U || true)

    if [ -z "$CONFLICTED_FILES" ]; then
        echo "✅ All conflicts were auto-resolved by rerere!"
        echo "Continuing rebase..."
        echo ""

        git -c core.editor=true rebase --continue

        # Check if rebase completed successfully
        if [ ! -d ".git/rebase-merge" ] && [ ! -d ".git/rebase-apply" ]; then
            echo ""
            echo "✅ Rebase completed successfully!"

            if [ "$GITHUB_ACTIONS_MODE" = "--github-actions" ]; then
                echo "status=success" >> $GITHUB_OUTPUT
                echo "has_conflicts=false" >> $GITHUB_OUTPUT
            fi

            exit 0
        else
            echo ""
            echo "⚠️ More conflicts detected after continue"

            if [ "$GITHUB_ACTIONS_MODE" = "--github-actions" ]; then
                echo "status=partial" >> $GITHUB_OUTPUT
                echo "has_conflicts=true" >> $GITHUB_OUTPUT
            fi

            echo "Conflicted files:"
            git diff --name-only --diff-filter=U

            exit 1
        fi
    else
        echo "⚠️ Unresolved conflicts detected!"
        echo ""

        if [ "$GITHUB_ACTIONS_MODE" = "--github-actions" ]; then
            echo "status=failed" >> $GITHUB_OUTPUT
            echo "has_conflicts=true" >> $GITHUB_OUTPUT
        fi

        echo "Conflicted files:"
        echo "$CONFLICTED_FILES"

        exit 1
    fi
fi
