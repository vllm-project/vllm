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

Archive hashes cover the package versions emitted in the five compiled locks.
Outside the documentation install, uv verifies supplied hashes but does not
require every installed package to have one. Coverage excludes:

- Versions replaced by overrides. For example, the CUDA image's `NCCL_VERSION`
  sets `UV_OVERRIDE` to install `nvidia-nccl-cu13==2.31.2`, replacing the lock's
  `2.30.7` pin. uv drops the original pin's hashes when applying that override,
  so the installed NCCL archive is not hash-verified against the lock.
- Packages from handwritten requirements such as `requirements/{common,cuda,rocm}.txt`
  and separate lint/build/bootstrap installs, unless the installed versions are
  also constrained by hashed lock entries.
- Packages named directly in Dockerfiles, such as `quart`, `msgpack`, and
  `aiohttp` in `Dockerfile.rocm`, unless covered by hashed lock entries in that
  install, and packages excluded from the ROCm lock.
- Source-built wheels and separate build dependencies. Hashing a source archive
  does not hash the resulting wheel or its build dependencies.

Globally requiring hashes needs complete locks for every dependency in the
install. Keeping NCCL hash-covered also requires updating the source requirements
and compiled lock together with future `NCCL_VERSION` changes.

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
