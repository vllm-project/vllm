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

## Reviewing an update

Automatic publishing is disabled by default. Download the patch artifact and
its recorded base SHA, check out that base, and apply the patch with
`git apply --index requirements.patch`. Review package versions and hashes
together before opening the update PR and running the normal compatibility
checks. The artifact only contains changes to the five compiled locks.

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
