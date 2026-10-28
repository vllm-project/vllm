# Dependency Hash Pinning

The compiled requirements in `requirements/docs.txt` and
`requirements/test/{cpu,cuda,rocm,xpu}.txt` pin package versions and SHA-256
archive hashes. Their pre-commit hooks use `uv pip compile --generate-hashes`;
run `pre-commit run --all-files` after changing source requirements. These
hooks preserve existing version pins unless a source change requires a new
resolution.

uv checks supplied hashes when downloading packages. Read the Docs additionally
passes `--require-hashes`, so missing hashes also fail its install. A hash
mismatch requires investigation; do not disable verification to complete an
install.

## Hash coverage

Archive hashes cover the packages emitted in the five compiled locks. Packages
excluded from the ROCm lock, separate lint/build/bootstrap installs, and
source-built wheels are outside that coverage. Hashing a source archive does
not hash its separate build dependencies. Mixed installs continue to use uv's
verification of supplied hashes; globally requiring hashes needs complete locks
for every dependency in the install.

## GitHub Actions commit pins

External actions and reusable workflows must use a full 40-character commit
SHA. Keep a version comment beside the SHA to make updates easy to review.
Dependabot can update those pins without replacing them with tags. Docker actions must use a
SHA-256 image digest; local `./` actions use the checked-out repository.

The `check-github-actions-pins` pre-commit hook rejects mutable tags, branches,
and abbreviated commits. To check all workflows directly, run:

```bash
pre-commit run check-github-actions-pins --all-files
```
