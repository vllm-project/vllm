#!/bin/bash
#
# cherry-pick-from-milestone.sh
# Find commits from a GitHub milestone that are missing from the current branch
# and output them in chronological order for cherry-picking.
#
# Usage: ./cherry-pick-from-milestone.sh <milestone> [--dry-run] [--execute]
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $(basename "$0") <milestone> [options]

Find commits from a GitHub milestone that need to be cherry-picked into the current branch.

Arguments:
    milestone       The GitHub milestone name (e.g., v0.14.0)

Options:
    --dry-run       Show the cherry-pick commands without executing (default)
    --execute       Actually execute the cherry-picks
    --main-branch   Specify the main branch name (default: main)
    --help          Show this help message

Examples:
    $(basename "$0") v0.14.0
    $(basename "$0") v0.14.0 --dry-run
    $(basename "$0") v0.14.0 --execute
    $(basename "$0") v0.14.0 --main-branch master
EOF
    exit 1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Default values
MILESTONE=""
DRY_RUN=true
MAIN_BRANCH="main"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --execute)
            DRY_RUN=false
            shift
            ;;
        --main-branch)
            MAIN_BRANCH="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        -*)
            log_error "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$MILESTONE" ]]; then
                MILESTONE="$1"
            else
                log_error "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Validate milestone argument
if [[ -z "$MILESTONE" ]]; then
    log_error "Milestone is required"
    usage
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    log_error "Not in a git repository"
    exit 1
fi

# Check if gh CLI is available
if ! command -v gh &>/dev/null; then
    log_error "GitHub CLI (gh) is not installed"
    exit 1
fi

# Check if authenticated with gh
if ! gh auth status &>/dev/null; then
    log_error "Not authenticated with GitHub CLI. Run 'gh auth login' first."
    exit 1
fi

CURRENT_BRANCH=$(git branch --show-current)
log_info "Current branch: ${CURRENT_BRANCH}"
log_info "Main branch: ${MAIN_BRANCH}"
log_info "Milestone: ${MILESTONE}"
echo ""

# Fetch latest from remote
log_info "Fetching latest from remote..."
git fetch origin "$MAIN_BRANCH" --quiet

# Get merged PRs from the milestone, sorted by merge date
log_info "Fetching merged PRs from milestone '${MILESTONE}'..."

# Store PR data in a temp file
PR_DATA=$(mktemp)
trap "rm -f $PR_DATA" EXIT

if ! gh pr list --state merged --search "milestone:${MILESTONE}" \
    --limit 1000 \
    --json number,title,mergeCommit,mergedAt \
    --jq 'sort_by(.mergedAt) | .[] | "\(.mergeCommit.oid)\t\(.number)\t\(.title)"' > "$PR_DATA" 2>/dev/null; then
    log_error "Failed to fetch PRs from milestone '${MILESTONE}'"
    log_error "This could be due to:"
    log_error "  - Milestone does not exist"
    log_error "  - Network/authentication issues"
    log_error "  - Invalid milestone name format"
    exit 1
fi

if [[ ! -s "$PR_DATA" ]]; then
    log_warn "No merged PRs found for milestone '${MILESTONE}'"
    exit 0
fi

TOTAL_PRS=$(wc -l < "$PR_DATA")
log_info "Found ${TOTAL_PRS} merged PR(s) in milestone"
echo ""

# Find commits that are missing from current branch
MISSING_COMMITS=()
MISSING_INFO=()

while IFS=$'\t' read -r sha pr_number title; do
    # Skip if SHA is empty or null
    if [[ -z "$sha" || "$sha" == "null" ]]; then
        log_warn "PR #${pr_number} has no merge commit SHA, skipping"
        continue
    fi
    
    # Check if this commit is already in the current branch
    if git merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
        log_success "PR #${pr_number} already in branch: ${title:0:60}"
    else
        log_warn "PR #${pr_number} MISSING: ${title:0:60}"
        MISSING_COMMITS+=("$sha")
        MISSING_INFO+=("$sha PR #${pr_number}: ${title}")
    fi
done < "$PR_DATA"

echo ""

if [[ ${#MISSING_COMMITS[@]} -eq 0 ]]; then
    log_success "All PRs from milestone '${MILESTONE}' are already in the current branch!"
    exit 0
fi

log_info "Found ${#MISSING_COMMITS[@]} missing commit(s) to cherry-pick"
echo ""

# Output the cherry-pick commands
echo "=========================================="
echo "Cherry-pick commands (in chronological order):"
echo "=========================================="
echo ""

for info in "${MISSING_INFO[@]}"; do
    echo "# $info"
done
echo ""

echo "# Run these commands to cherry-pick all missing commits:"
echo "git cherry-pick ${MISSING_COMMITS[*]}"
echo ""

# Or one by one
echo "# Or cherry-pick one at a time:"
for sha in "${MISSING_COMMITS[@]}"; do
    echo "git cherry-pick $sha"
done
echo ""

# Execute if requested
if [[ "$DRY_RUN" == false ]]; then
    echo "=========================================="
    log_info "Executing cherry-picks..."
    echo "=========================================="
    
    for i in "${!MISSING_COMMITS[@]}"; do
        sha="${MISSING_COMMITS[$i]}"
        info="${MISSING_INFO[$i]}"
        
        echo ""
        log_info "Cherry-picking: $info"
        
        if git cherry-pick "$sha"; then
            log_success "Successfully cherry-picked $sha"
        else
            log_error "Failed to cherry-pick $sha"
            log_error "Resolve conflicts and run 'git cherry-pick --continue', or 'git cherry-pick --abort' to cancel"
            exit 1
        fi
    done
    
    echo ""
    log_success "All cherry-picks completed successfully!"
else
    echo "=========================================="
    echo -e "${YELLOW}Dry run mode - no changes made${NC}"
    echo "Run with --execute to perform the cherry-picks"
    echo "=========================================="
fi
