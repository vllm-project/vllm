# Automatic Dependency Updates

The **Refresh compiled requirements** workflow runs weekly on Monday at
07:00 UTC and supports manual dispatch on `main`. It upgrades
`requirements/docs.txt` and `requirements/test/{cpu,cuda,rocm,xpu}.txt` within
their source constraints, including the corresponding SHA-256 archive hashes.
It then normalizes each output with its pre-commit command, so ordinary compile
hooks keep the updated locks unchanged.

The updater requires all five pre-commit compile hooks to use
`--generate-hashes`; it stops if any hook lacks that flag. Hash initialization,
install verification, and enforcement checks are supplied separately from this
workflow. The updater itself does not change the compile hooks or initialize
package hashes on installation.

## Seven-day cooldown

Routine Python refreshes wait at least seven days after each archive's upload time.
The helper computes one UTC cutoff and applies it to both the upgrade and
normalization passes through `UV_EXCLUDE_NEWER`, keeping generated command
headers compatible with pre-commit. The workflow also applies a seven-day
cutoff when installing its PyYAML helper dependency.

Archives without upload timestamps are excluded. If source constraints require
an archive that is too new or has no timestamp, resolution fails and no review
artifact is published. This policy requires timestamp support in the package
index; it does not apply to Git, local, or direct-URL dependencies.

The XPU index currently omits upload timestamps for required wheels, including
`triton==3.8.0+xpu`. With hash-generating hooks in place, the complete five-lock
workflow stops at XPU; that index must supply timestamps before the refresh can
succeed. No package or index is exempted to bypass the cooldown.

Dependabot uses `cooldown.default-days: 7` for routine GitHub Actions and Cargo
updates. Its Python security updates remain exempt from cooldown, so reviewed
security fixes can proceed promptly. Urgent fixes that cannot wait for the
compiled refresh need a separately reviewed manual update.

See [uv's upload cutoff documentation](https://docs.astral.sh/uv/concepts/resolution/#reproducible-resolution)
and [Dependabot's cooldown options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference#cooldown).

Cooldown gives the ecosystem time to detect malicious releases. It does not
establish package safety or publisher provenance; review those separately from
the archive hashes.

## Reviewing an update

After compilation, the workflow runs `tools/review_compiled_requirements.py`
against the recorded base commit. It checks every pin in all five locks against
the [OSV advisory API](https://google.github.io/osv.dev/post-v1-querybatch/),
including unchanged dependencies and all environment markers, without installing
or importing the target packages. For local vendor versions such as `+xpu`, it
also queries the upstream public version to catch inherited advisories.

**Vulnerability blocking is disabled by default.** Both existing and newly
introduced vulnerabilities are included for human review and do not block the
refresh or patch publication. Network/service errors, malformed or incomplete
responses, and paginated results are reported as incomplete checks and also do
not block publication. Unsupported pins, missing hashes, and empty locks are
reported as coverage errors rather than silently treated as safe. The workflow's
reporting step uses `continue-on-error: true`, so even a reporting-tool failure
does not prevent publication of the compiled patch.

The job summary shows the available review. Artifacts include
`dependency-review.md` and `dependency-review.json` alongside the patch and base
SHA when report generation succeeds. The report lists added/removed dependencies,
version changes, and added/removed hash counts, highlighting hash changes at the
same version. The JSON contains the complete before/after hashes for each change,
per-lock parsing counts, every completed package/version query with its advisory
IDs, and any coverage errors. Check the JSON `complete` field and the job logs
before interpreting a report without findings; missing or incomplete reports do
not indicate that the dependencies have no known vulnerabilities.

To generate the same informational report locally after regenerating locks:

```bash
uv run --no-project --exclude-newer '7 days' --with packaging==25.0 \
  --with regex==2026.2.28 --python 3.12 \
  python tools/review_compiled_requirements.py \
  --base <commit-before-refresh> --output-dir /tmp/dependency-review
```

The helper's optional `--enforce` flag returns a failing exit status for findings
or incomplete checks. The scheduled workflow does not pass this flag. Enforcing
it in that workflow would require deliberately adding the flag and removing
`continue-on-error: true` from the reporting step.

The report checks known advisories by package name and version; it cannot detect
undisclosed malicious code or establish publisher provenance. Vendor forks may
differ from upstream, and an empty advisory result does not certify a package.
The locks do not record artifact sources, so this report cannot detect source
changes. Review configured indexes and upstream provenance separately.

Automatic publishing is disabled by default. Download the patch artifact and
its recorded base SHA, check out that base, and apply the patch with
`git apply --index requirements.patch`. Read both review reports, inspect new
dependencies and same-version hash changes, and review package versions and
hashes together before opening the update PR and running the normal compatibility
checks. The patch only contains changes to the five compiled locks.

The compile job has read-only repository access and no persisted checkout
credentials. Enabling automatic publishing later requires configuring the
repository-scoped GitHub App and deliberately enabling the separate publish
job. That job validates the artifact before obtaining write credentials and
accepts only regular-file modifications to the five compiled locks.

## Dependabot coverage

GitHub Actions updates are grouped weekly, as are Rust minor/patch updates.
Dependabot provides security-only updates for handwritten Python requirements,
including build and connector manifests; the five compiled outputs are handled
by the platform-aware refresh workflow instead. Source-manifest updates still
need the normal compile hooks, compatibility testing, and human review.

A compiled lock refresh does not upgrade an incompatible exact source pin or
fix an advisory without a patched release. It does not upgrade separate build
or connector pins. These cases still need manual remediation.
