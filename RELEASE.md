# Releasing vLLM

vLLM releases offer a reliable version of the code base, packaged into a binary format that can be conveniently accessed via PyPI. These releases also serve as key milestones for the development team to communicate with the community about newly available features, improvements, and upcoming changes that could affect users, including potential breaking changes.

## Release Versioning

vLLM uses a “right-shifted” versioning scheme where a new patch release is out every 2 weeks. And patch releases contain features and bug fixes (as opposed to semver where patch release contains only backwards-compatible bug fixes). When critical fixes need to be made, special release post1 is released.

* _major_ major architectural milestone and when incompatible API changes are made, similar to PyTorch 2.0.
* _minor_ major features
* _patch_ features and backwards-compatible bug fixes
* _post1_ or _patch-1_ backwards-compatible bug fixes, either explicit or implicit post release

## Release Cadence

Patch release is released on bi-weekly basis. Post release 1-3 days after patch release and uses same branch as patch release.
Following is the release cadence for year 2025. All future release dates below are tentative. Please note: Post releases are optional.

| Release Date | Patch release versions | Post Release versions |
| --- | --- | --- |
| Jan 2025 | 0.7.0 | --- |
| Feb 2025 | 0.7.1, 0.7.2, 0.7.3  | --- |
| Mar 2025 | 0.7.4, 0.7.5 | --- |
| Apr 2025 | 0.7.6, 0.7.7 | --- |
| May 2025 | 0.7.8, 0.7.9 | --- |
| Jun 2025 | 0.7.10, 0.7.11 | --- |
| Jul 2025 | 0.7.12, 0.7.13 | --- |
| Aug 2025 | 0.7.14, 0.7.15 | --- |
| Sep 2025 | 0.7.16, 0.7.17 | --- |
| Oct 2025 | 0.7.18, 0.7.19 | --- |
| Nov 2025 | 0.7.20, 0.7.21 | --- |
| Dec 2025 | 0.7.22, 0.7.23 | --- |

## Release branch

Each release is built from a dedicated release branch.

* For _major_, _minor_, _patch_ releases, the release branch cut is performed 1-2 days before release is live.
* For post releases, previously cut release branch is reused
* Release builds are triggered via push to RC tag like vX.Y.Z-rc1 . This enables us to build and test multiple RCs for each release.
* Final tag : vX.Y.Z does not trigger the build but used for Release notes and assets.
* After branch cut is created we monitor the main branch for any reverts and apply these reverts to a release branch.

## Release Cherry-Pick Criteria

After branch cut, we approach finalizing the release branch with clear criteria on what cherry picks are allowed in. Note: a cherry pick is a process to land a PR in the release branch after branch cut. These are typically limited to ensure that the team has sufficient time to complete a thorough round of testing on a stable code base.

* Regression fixes - that address functional/performance regression against the most recent release (e.g. 0.7.0 for 0.7.1 release)
* Critical fixes - critical fixes for severe issue such as silent incorrectness, backwards compatibility, crashes, deadlocks, (large) memory leaks
* Fixes to new features introduced in the most recent release (e.g. 0.7.0 for 0.7.1 release)
* Documentation improvements
* Release branch specific changes (e.g. change version identifiers or CI fixes)

Please note: **No feature work allowed for cherry picks**. All PRs that are considered for cherry-picks need to be merged on trunk, the only exception are Release branch specific changes.
