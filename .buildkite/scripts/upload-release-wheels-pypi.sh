#!/usr/bin/env bash

set -e

BUCKET="vllm-wheels"
SUBPATH=$BUILDKITE_COMMIT
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

RELEASE_VERSION=$(buildkite-agent meta-data get release-version)
GIT_VERSION=$(git describe --exact-match --tags $BUILDKITE_COMMIT 2>/dev/null)

echo "Release version from Buildkite: $RELEASE_VERSION"

if [[ -z "$GIT_VERSION" ]]; then
    echo "[FATAL] Not on a git tag, cannot create release."
    exit 1
else
    echo "Git version for commit $BUILDKITE_COMMIT: $GIT_VERSION"
fi
# sanity check for version mismatch
if [[ "$RELEASE_VERSION" != "$GIT_VERSION" ]]; then
  if [[ "$FORCE_RELEASE_IGNORE_VERSION_MISMATCH" == "true" ]]; then
    echo "[WARNING] Force release and ignore version mismatch"
  else
    echo "[FATAL] Release version from Buildkite does not match Git version."
    exit 1
  fi
fi
PURE_VERSION=${RELEASE_VERSION#v} # remove leading 'v'

# check pypi token
if [[ -z "$PYPI_TOKEN" ]]; then
  echo "[FATAL] PYPI_TOKEN is not set."
  exit 1
else
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="$PYPI_TOKEN"
fi

set -x # avoid printing secrets above

# install twine from pypi
python3 -m venv /tmp/vllm-release-env
source /tmp/vllm-release-env/bin/activate
pip install twine
python3 -m twine --version

# copy release wheels to local directory
DIST_DIR=/tmp/vllm-release-dist
echo "Existing wheels on S3:"
aws s3 ls "$S3_COMMIT_PREFIX"
echo "Copying wheels to local directory"
mkdir -p $DIST_DIR
# include only wheels for the release version, ignore all files with "dev" or "rc" in the name (without excluding 'aarch64')
aws s3 cp --recursive --exclude "*" --include "vllm-${PURE_VERSION}*.whl" --exclude "*dev*" --exclude "*rc[0-9]*" "$S3_COMMIT_PREFIX" $DIST_DIR
echo "Wheels copied to local directory"
# generate source tarball
git archive --format=tar.gz --output="$DIST_DIR/vllm-${PURE_VERSION}.tar.gz" $BUILDKITE_COMMIT
ls -la $DIST_DIR

# upload wheels to PyPI (only default variant, i.e. files without '+' in the name)
PYPI_WHEEL_FILES=$(find $DIST_DIR -name "vllm-${PURE_VERSION}*.whl" -not -name "*+*")
if [[ -z "$PYPI_WHEEL_FILES" ]]; then
  echo "No default variant wheels found, quitting..."
  exit 1
fi

python3 -m twine check $PYPI_WHEEL_FILES
python3 -m twine upload --non-interactive --verbose $PYPI_WHEEL_FILES
echo "Wheels uploaded to PyPI"
