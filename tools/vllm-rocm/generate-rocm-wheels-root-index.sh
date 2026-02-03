#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Generate S3 PyPI Root Index for Latest Version
#
# Creates a PEP 503 compatible index.html at rocm/ pointing to the latest
# semantic version's packages. This enables users to install with:
#   uv pip install vllm --extra-index-url s3://vllm-wheels/rocm
#
# Usage:
#   generate-root-index.sh [options]
#
# Options:
#   --dry-run      Preview changes without uploading
#   --version VER  Use specific version instead of auto-detecting latest
#
# Environment variables:
#   S3_BUCKET   - Bucket name (default: vllm-wheels)
#   VARIANT     - ROCm variant (default: rocm700)
#   DRY_RUN     - Set to 1 for preview mode (same as --dry-run)

set -euo pipefail

# ======== Configuration ========
BUCKET="${S3_BUCKET:-vllm-wheels}"
VARIANT="${VARIANT:-rocm700}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_VERSION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --version)
            FORCE_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Working directory for generated files
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "========================================"
echo "Generate Root Index for Latest Version"
echo "========================================"
echo "S3 Bucket: $BUCKET"
echo "ROCm Variant: $VARIANT"
echo "Dry Run: $DRY_RUN"
echo "========================================"
echo ""

# ======== Step 1: Find latest semantic version ========

echo "Step 1: Finding latest semantic version..."

# List all directories under rocm/
aws s3api list-objects-v2 \
    --bucket "$BUCKET" \
    --prefix "rocm/" \
    --delimiter "/" \
    --query 'CommonPrefixes[].Prefix' \
    --output text | tr '\t' '\n' > "$WORK_DIR/all_prefixes.txt"

# Filter for semantic versions (x.y.z pattern)
grep -oE 'rocm/[0-9]+\.[0-9]+\.[0-9]+/' "$WORK_DIR/all_prefixes.txt" | \
    sed 's|rocm/||; s|/||' | \
    sort -V > "$WORK_DIR/versions.txt" || true

if [[ ! -s "$WORK_DIR/versions.txt" ]]; then
    echo "ERROR: No semantic versions found under s3://$BUCKET/rocm/"
    exit 1
fi

echo "Found versions:"
cat "$WORK_DIR/versions.txt"
echo ""

if [[ -n "$FORCE_VERSION" ]]; then
    LATEST_VERSION="$FORCE_VERSION"
    echo "Using forced version: $LATEST_VERSION"
else
    LATEST_VERSION=$(tail -1 "$WORK_DIR/versions.txt")
    echo "Latest version (auto-detected): $LATEST_VERSION"
fi

# Verify the version exists
if ! grep -qx "$LATEST_VERSION" "$WORK_DIR/versions.txt"; then
    echo "ERROR: Version $LATEST_VERSION not found in bucket"
    exit 1
fi

# ======== Step 2: List packages from latest version ========

echo ""
echo "Step 2: Listing packages from rocm/$LATEST_VERSION/$VARIANT/..."

VERSION_PREFIX="rocm/$LATEST_VERSION/$VARIANT/"

# List package directories
aws s3api list-objects-v2 \
    --bucket "$BUCKET" \
    --prefix "$VERSION_PREFIX" \
    --delimiter "/" \
    --query 'CommonPrefixes[].Prefix' \
    --output text | tr '\t' '\n' > "$WORK_DIR/package_prefixes.txt" || true

if [[ ! -s "$WORK_DIR/package_prefixes.txt" ]]; then
    echo "ERROR: No packages found under s3://$BUCKET/$VERSION_PREFIX"
    exit 1
fi

# Extract package names
sed "s|${VERSION_PREFIX}||; s|/||g" "$WORK_DIR/package_prefixes.txt" | \
    grep -v '^$' > "$WORK_DIR/packages.txt"

echo "Found packages:"
cat "$WORK_DIR/packages.txt"
echo ""

# ======== Step 3: Generate root index.html ========

echo "Step 3: Generating root index.html..."

mkdir -p "$WORK_DIR/output"

{
    cat <<'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta name="pypi:repository-version" content="1.0">
</head>
<body>
EOF

    while read -r pkg; do
        echo "    <a href=\"$pkg/\">$pkg</a><br>"
    done < "$WORK_DIR/packages.txt"

    cat <<'EOF'
</body>
</html>
EOF
} > "$WORK_DIR/output/index.html"

echo "Generated root index.html:"
cat "$WORK_DIR/output/index.html"
echo ""

# ======== Step 4: Copy and adjust package index files ========

echo "Step 4: Copying and adjusting package index files..."

while read -r pkg; do
    echo "Processing package: $pkg"

    # Download existing index.html from versioned path
    SOURCE_INDEX="s3://$BUCKET/$VERSION_PREFIX$pkg/index.html"

    mkdir -p "$WORK_DIR/output/$pkg"

    if aws s3 cp "$SOURCE_INDEX" "$WORK_DIR/output/$pkg/index.html" 2>/dev/null; then
        # Adjust relative paths:
        # Original: href="../../../{commit}/wheel.whl" (from rocm/0.13.0/rocm710/vllm/)
        # New:      href="../{commit}/wheel.whl"       (from rocm/vllm/)
        sed -i 's|href="\.\./\.\./\.\./|href="../|g' "$WORK_DIR/output/$pkg/index.html"
        echo "  - Downloaded and adjusted: $pkg/index.html"
    else
        echo "  - WARNING: Could not download index for $pkg"
    fi
done < "$WORK_DIR/packages.txt"

echo ""

# ======== Step 5: Upload to S3 ========

echo "Step 5: Uploading to s3://$BUCKET/rocm/..."
echo ""

# List what would be uploaded
echo "Files to upload:"
find "$WORK_DIR/output" -name "*.html" -type f | while read -r file; do
    rel_path="${file#$WORK_DIR/output/}"
    echo "  rocm/$rel_path"
done
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN - Skipping upload"
    echo ""
    echo "Preview of generated files:"
    echo "----------------------------------------"
    echo "rocm/index.html:"
    cat "$WORK_DIR/output/index.html"
    echo ""
    echo "----------------------------------------"
    echo "Sample package index (first package):"
    FIRST_PKG=$(head -1 "$WORK_DIR/packages.txt")
    if [[ -f "$WORK_DIR/output/$FIRST_PKG/index.html" ]]; then
        echo "rocm/$FIRST_PKG/index.html:"
        cat "$WORK_DIR/output/$FIRST_PKG/index.html"
    fi
else
    # Upload all generated files
    aws s3 cp --recursive "$WORK_DIR/output/" "s3://$BUCKET/rocm/" \
        --content-type "text/html"

    echo "Upload complete!"
fi

# ======== Summary ========

echo ""
echo "========================================"
echo "Root Index Generation Complete!"
echo "========================================"
echo ""
echo "Latest version: $LATEST_VERSION"
echo "Packages indexed: $(wc -l < "$WORK_DIR/packages.txt")"
echo ""
echo "Install command:"
echo "  uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/"
echo "========================================"
