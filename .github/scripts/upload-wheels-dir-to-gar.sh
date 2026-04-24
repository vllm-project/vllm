#!/usr/bin/env bash
set -euxo pipefail

# Script to upload Python wheels from a directory to Google Artifact Registry
# Usage: ./upload-wheels-dir-to-gar.sh <wheels-dir> <provider>
# Example: ./upload-wheels-dir-to-gar.sh ./wheels amd

WHEELS_DIR="${1:?Error: Wheels directory required}"
PROVIDER="${2:?Error: Provider required (nvidia or amd)}"

# Configuration
GAR_LOCATION="us-central1"
GAR_PROJECT="cohere-artifacts"
GAR_REPOSITORY="cohere-py-forks-${PROVIDER}"
REPOSITORY_URL="https://${GAR_LOCATION}-python.pkg.dev/${GAR_PROJECT}/${GAR_REPOSITORY}/"

echo "========================================="
echo "Uploading wheels to Google Artifact Registry"
echo "Directory: $WHEELS_DIR"
echo "Provider: $PROVIDER"
echo "========================================="

# Check if directory exists and contains wheels
if [ ! -d "$WHEELS_DIR" ]; then
    echo "Error: Directory $WHEELS_DIR does not exist"
    exit 1
fi

WHEEL_COUNT=$(find "$WHEELS_DIR" -name "*.whl" | wc -l)
if [ "$WHEEL_COUNT" -eq 0 ]; then
    echo "Error: No wheel files found in $WHEELS_DIR"
    exit 1
fi

echo "Found $WHEEL_COUNT wheel file(s)"
ls -lh "$WHEELS_DIR"/*.whl

# Configure gcloud for artifact registry
echo "Configuring gcloud for Artifact Registry..."
gcloud config set project "$GAR_PROJECT"

# Ensure the Python repository exists
echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe "$GAR_REPOSITORY" \
    --location="$GAR_LOCATION" \
    --project="$GAR_PROJECT" &>/dev/null || \
    gcloud artifacts repositories create "$GAR_REPOSITORY" \
    --location="$GAR_LOCATION" \
    --repository-format=python \
    --project="$GAR_PROJECT" \
    --description="Python wheels for vLLM ${PROVIDER} builds"

# Install twine and keyring for uploading
echo "Installing twine and Google Artifact Registry auth..."
python3 -m pip install --upgrade pip
python3 -m pip install --force-reinstall --no-cache-dir 'twine>=5.1.0' keyrings.google-artifactregistry-auth

# Verify twine version
echo "Installed twine version:"
python3 -m twine --version

# Upload wheels using twine (one at a time to handle partial failures)
echo "========================================="
echo "Uploading to: $REPOSITORY_URL"
echo "========================================="

# Upload each wheel individually so that already-uploaded wheels
# (HTTP 409 conflict) don't cause the entire upload to fail.
# Note: --skip-existing is not supported by Google Artifact Registry,
# so we handle conflicts manually.
FAILED=0
SKIPPED=0
UPLOADED=0
for whl in "$WHEELS_DIR"/*.whl; do
    echo "-----------------------------------------"
    echo "Uploading: $(basename "$whl")"
    if UPLOAD_OUTPUT=$(python3 -m twine upload \
        --repository-url "$REPOSITORY_URL" \
        --verbose \
        "$whl" 2>&1); then
        echo "$UPLOAD_OUTPUT"
        echo "✅ Uploaded: $(basename "$whl")"
        UPLOADED=$((UPLOADED + 1))
    else
        # Upload failed — check if it was a conflict (package already exists)
        echo "$UPLOAD_OUTPUT"
        if echo "$UPLOAD_OUTPUT" | grep -qiE "409|conflict|already exists"; then
            echo "⚠️  Already exists, skipping: $(basename "$whl")"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "❌ Failed to upload: $(basename "$whl")"
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo "========================================="
echo "Upload summary: $UPLOADED uploaded, $SKIPPED already existed, $FAILED failed"
if [ "$FAILED" -gt 0 ]; then
    echo "❌ $FAILED wheel(s) failed to upload"
    exit 1
fi
echo "✅ All wheels uploaded successfully!"
echo "Install with: pip install vllm --extra-index-url https://${GAR_LOCATION}-python.pkg.dev/${GAR_PROJECT}/${GAR_REPOSITORY}/simple/"
echo "========================================="
