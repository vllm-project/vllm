#!/bin/bash
#
# trigger-ci-build.sh
# Trigger a Buildkite CI build using the bk CLI for the current commit and branch
# with RUN_ALL=1 and NIGHTLY=1 environment variables.
#
# Usage: ./trigger-ci-build.sh [options]
#
# Requires: bk CLI (https://buildkite.com/docs/platform/cli)
#
# SAFETY: Dry-run by default. Use --execute to actually trigger a build.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
PIPELINE="ci"
DRY_RUN=true

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Trigger a Buildkite CI build using the bk CLI for the current commit and branch.
Sets RUN_ALL=1 and NIGHTLY=1 environment variables.

SAFETY: Dry-run by default. Use --execute to actually trigger a build.

Options:
    --execute       Actually trigger the build (default: dry-run)
    --pipeline      Buildkite pipeline slug (default: ${PIPELINE})
    --commit        Override commit SHA (default: current HEAD)
    --branch        Override branch name (default: current branch)
    --message       Custom build message (default: auto-generated)
    --help          Show this help message

Prerequisites:
    - bk CLI installed: brew tap buildkite/buildkite && brew install buildkite/buildkite/bk
    - bk configured: bk configure

Examples:
    $(basename "$0")                        # Dry-run, show what would happen
    $(basename "$0") --execute              # Actually trigger the build
    $(basename "$0") --pipeline ci-shadow   # Dry-run with different pipeline
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

# Parse arguments
COMMIT=""
BRANCH=""
MESSAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --execute)
            DRY_RUN=false
            shift
            ;;
        --pipeline)
            PIPELINE="$2"
            shift 2
            ;;
        --commit)
            COMMIT="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --message)
            MESSAGE="$2"
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
            log_error "Unexpected argument: $1"
            usage
            ;;
    esac
done

# Check if bk CLI is installed
if ! command -v bk &>/dev/null; then
    log_error "Buildkite CLI (bk) is not installed"
    echo ""
    echo "Install with:"
    echo "  brew tap buildkite/buildkite && brew install buildkite/buildkite/bk"
    echo ""
    echo "Then configure:"
    echo "  bk configure"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    log_error "Not in a git repository"
    exit 1
fi

# Get current commit and branch if not overridden
if [[ -z "$COMMIT" ]]; then
    COMMIT=$(git rev-parse HEAD)
fi

if [[ -z "$BRANCH" ]]; then
    BRANCH=$(git branch --show-current)
    if [[ -z "$BRANCH" ]]; then
        # Detached HEAD state - try to get branch from ref
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
    fi
fi

# Generate default message if not provided
if [[ -z "$MESSAGE" ]]; then
    COMMIT_MSG=$(git log -1 --pretty=format:"%s" "$COMMIT" 2>/dev/null || echo "Manual build")
    MESSAGE="[Manual] ${COMMIT_MSG}"
fi

# Safety check: Verify the commit exists on the remote
log_info "Verifying commit exists on remote..."
git fetch origin --quiet 2>/dev/null || true

# Check if commit is reachable from any remote branch
REMOTE_BRANCHES=$(git branch -r --contains "$COMMIT" 2>/dev/null || true)
if [[ -z "$REMOTE_BRANCHES" ]]; then
    log_error "Commit ${COMMIT} does not exist on any remote branch!"
    echo ""
    echo "The CI system will fail to checkout this commit."
    echo "Please push your changes first:"
    echo ""
    echo "  git push origin ${BRANCH}"
    echo ""
    exit 1
fi

log_success "Commit found on remote branches:"
echo "$REMOTE_BRANCHES" | head -5 | sed 's/^/  /'
if [[ $(echo "$REMOTE_BRANCHES" | wc -l) -gt 5 ]]; then
    echo "  ... and more"
fi
echo ""

log_info "Pipeline: ${PIPELINE}"
log_info "Branch: ${BRANCH}"
log_info "Commit: ${COMMIT}"
log_info "Message: ${MESSAGE}"
log_info "Environment: RUN_ALL=1, NIGHTLY=1"
echo ""

# Build the command
CMD=(bk build create
    -y
    -w
    -i
    --pipeline "${PIPELINE}"
    --commit "${COMMIT}"
    --branch "${BRANCH}"
    --message "${MESSAGE}"
    --env "RUN_ALL=1"
    --env "NIGHTLY=1"
)

if [[ "$DRY_RUN" == true ]]; then
    echo "=========================================="
    log_warn "DRY-RUN MODE - No build will be triggered"
    echo "=========================================="
    echo ""
    echo "Command that would be executed:"
    echo ""
    # Escape single quotes in values for safe shell display
    escape_for_shell() {
        printf '%s' "$1" | sed "s/'/'\\\\''/g"
    }
    echo "  bk build create \\"
    echo "    -y \\"
    echo "    -w \\"
    echo "    -i \\"
    echo "    --pipeline '$(escape_for_shell "${PIPELINE}")' \\"
    echo "    --commit '$(escape_for_shell "${COMMIT}")' \\"
    echo "    --branch '$(escape_for_shell "${BRANCH}")' \\"
    echo "    --message '$(escape_for_shell "${MESSAGE}")' \\"
    echo "    --env 'RUN_ALL=1' \\"
    echo "    --env 'NIGHTLY=1'"
    echo ""
    echo "=========================================="
    echo -e "${YELLOW}To actually trigger this build, run:${NC}"
    echo ""
    echo "  $0 --execute"
    echo "=========================================="
    exit 0
fi

log_info "Triggering build..."

# Execute the command - bk will print the URL and open browser
"${CMD[@]}"
