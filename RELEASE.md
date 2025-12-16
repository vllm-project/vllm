# Releasing vLLM

vLLM releases offer a reliable version of the code base, packaged into a binary format that can be conveniently accessed via [PyPI](https://pypi.org/project/vllm). These releases also serve as key milestones for the development team to communicate with the community about newly available features, improvements, and upcoming changes that could affect users, including potential breaking changes.

## Release Cadence and Versioning

We aim to have a regular release every 2 weeks. Since v0.12.0, regular releases increment the minor version rather than patch version. The list of past releases can be found [here](https://vllm.ai/releases).

Our version numbers are expressed in the form `vX.Y.Z`, where `X` is the major version, `Y` is the minor version, and `Z` is the patch version. They are incremented according to the following rules:

* _Major_ releases are reserved for architectural milestones involving sweeping API changes, similar to PyTorch 2.0.
* _Minor_ releases correspond to regular releases, which include new features, bug fixes and other backwards-compatible changes.
* _Patch_ releases correspond to special releases for new models, as well as emergency patches for critical performance, functionality and security issues.

<<<<<<< HEAD
This versioning scheme is similar to [SemVer](https://semver.org/) for compatibility purposes, except that backwards compatibility is only guaranteed for a limited number of minor releases (see our [deprecation policy](https://docs.vllm.ai/en/latest/contributing/deprecation_policy) for details).
=======
Patch release is released on bi-weekly basis. Post release 1-2 days after patch release and uses same branch as patch release.
Following is the release cadence for year 2025. All future release dates below are tentative. Please note: Post releases are optional.
>>>>>>> ea8bbee26 (release_210_testing)

## Release Branch

Each release is built from a dedicated release branch.

* For _major_ and _minor_ releases, the release branch cut is performed 1-2 days before release is live.
* For _patch_ releases, previously cut release branch is reused.
* Release builds are triggered via push to RC tag like `vX.Y.Z-rc1`. This enables us to build and test multiple RCs for each release.
* Final tag: `vX.Y.Z` does not trigger the build but used for Release notes and assets.
* After branch cut is created, we monitor the main branch for any reverts and apply these reverts to a release branch.

### Cherry-Pick Criteria

After branch cut, we approach finalizing the release branch with clear criteria on what cherry picks are allowed in. Note: a cherry pick is a process to land a PR in the release branch after branch cut. These are typically limited to ensure that the team has sufficient time to complete a thorough round of testing on a stable code base.

* Regression fixes - that address functional/performance regression against the most recent release (e.g. 0.7.0 for 0.7.1 release)
* Critical fixes - critical fixes for severe issue such as silent incorrectness, backwards compatibility, crashes, deadlocks, (large) memory leaks
* Fixes to new features introduced in the most recent release (e.g. 0.7.0 for 0.7.1 release)
* Documentation improvements
* Release branch specific changes (e.g. change version identifiers or CI fixes)

Please note: **No feature work allowed for cherry picks**. All PRs that are considered for cherry-picks need to be merged on trunk, the only exception are Release branch specific changes.

## Manual validations

### E2E Performance Validation

Before each release, we perform end-to-end performance validation to ensure no regressions are introduced. This validation uses the [vllm-benchmark workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) on PyTorch CI.

**Current Coverage:**

* Models: Llama3, Llama4, and Mixtral
* Hardware: NVIDIA H100 and AMD MI300x
* _Note: Coverage may change based on new model releases and hardware availability_

**Performance Validation Process:**

**Step 1: Get Access**
Request write access to the [pytorch/pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing) repository to run the benchmark workflow.

**Step 2: Review Benchmark Setup**
Familiarize yourself with the benchmark configurations:

* [CUDA setup](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/cuda)
* [ROCm setup](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/rocm)

**Step 3: Run the Benchmark**
Navigate to the [vllm-benchmark workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) and configure:

* **vLLM branch**: Set to the release branch (e.g., `releases/v0.9.2`)
* **vLLM commit**: Set to the RC commit hash

**Step 4: Review Results**
Once the workflow completes, benchmark results will be available on the [vLLM benchmark dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm) under the corresponding branch and commit.

**Step 5: Performance Comparison**
Compare the current results against the previous release to verify no performance regressions have occurred. Here is an
example of [v0.9.1 vs v0.9.2](https://hud.pytorch.org/benchmark/llms?startTime=Thu%2C%2017%20Apr%202025%2021%3A43%3A50%20GMT&stopTime=Wed%2C%2016%20Jul%202025%2021%3A43%3A50%20GMT&granularity=week&lBranch=releases/v0.9.1&lCommit=b6553be1bc75f046b00046a4ad7576364d03c835&rBranch=releases/v0.9.2&rCommit=a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f&repoName=vllm-project%2Fvllm&benchmarkName=&modelName=All%20Models&backendName=All%20Backends&modeName=All%20Modes&dtypeName=All%20DType&deviceName=All%20Devices&archName=All%20Platforms).
