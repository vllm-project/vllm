#!/usr/bin/env bash

set -e

BUCKET="vllm-wheels"
SUBPATH=$BUILDKITE_COMMIT
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

RELEASE_VERSION=$(buildkite-agent meta-data get release-version)
echo "Release version from Buildkite: $RELEASE_VERSION"
GIT_VERSION=$(git describe --exact-match --tags $BUILDKITE_COMMIT 2>/dev/null)
if [ -z "$GIT_VERSION" ]; then
    echo "[FATAL] Not on a git tag, cannot create release."
    exit 1
else
    echo "Git version for commit $BUILDKITE_COMMIT: $GIT_VERSION"
fi
# sanity check for version mismatch
if [ "$RELEASE_VERSION" != "$GIT_VERSION" ]; then
  if [ "$FORCE_RELEASE_IGNORE_VERSION_MISMATCH" == "true" ]; then
    echo "[WARNING] Force release and ignore version mismatch"
  else
    echo "[FATAL] Release version from Buildkite does not match Git version."
    exit 1
  fi
fi
PURE_VERSION=${RELEASE_VERSION#v} # remove leading 'v'

# check pypi token
if [ -z "$PYPI_TOKEN" ]; then
  echo "[FATAL] PYPI_TOKEN is not set."
  exit 1
else
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="$PYPI_TOKEN"
fi

# check github token
if [ -z "$GITHUB_TOKEN" ]; then
  echo "[FATAL] GITHUB_TOKEN is not set."
  exit 1
else
  export GH_TOKEN="$GITHUB_TOKEN"
fi

set -x # avoid printing secrets above

# download gh CLI from github
# Get latest gh CLI version from GitHub API
GH_VERSION=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' | sed 's/^v//')
if [ -z "$GH_VERSION" ]; then
  echo "[FATAL] Failed to get latest gh CLI version from GitHub"
  exit 1
fi
echo "Downloading gh CLI version: $GH_VERSION"
GH_TARBALL="gh_${GH_VERSION}_linux_amd64.tar.gz"
GH_URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/${GH_TARBALL}"
GH_INSTALL_DIR="/tmp/gh-install"
mkdir -p "$GH_INSTALL_DIR"
pushd "$GH_INSTALL_DIR"
curl -L -o "$GH_TARBALL" "$GH_URL"
tar -xzf "$GH_TARBALL"
GH_BIN=$(realpath $(find . -name "gh" -type f -executable | head -n 1))
if [ -z "$GH_BIN" ]; then
  echo "[FATAL] Failed to find gh CLI executable"
  exit 1
fi
echo "gh CLI downloaded successfully, version: $($GH_BIN --version)"
echo "Last 5 releases on GitHub:" # as a sanity check of gh and GH_TOKEN
command "$GH_BIN" release list --limit 5
popd

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
if [ -z "$PYPI_WHEEL_FILES" ]; then
  echo "No default variant wheels found, quitting..."
  exit 1
fi
python3 -m twine check $PYPI_WHEEL_FILES
python3 -m twine --non-interactive --verbose upload $PYPI_WHEEL_FILES
echo "Wheels uploaded to PyPI"

# create release on GitHub with the release version and all wheels
command "$GH_BIN" release create $GIT_VERSION -d --latest --notes-from-tag --verify-tag $DIST_DIR/*.whl
