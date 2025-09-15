
## Description

This file contains the downloading link for benchmarking results.

- [benchmarking pipeline](artifact://nightly-pipeline.yaml)
- [benchmarking results](artifact://results.zip)
- [benchmarking code](artifact://nightly-benchmarks.zip)

Please download the visualization scripts in the post

## Results reproduction

- Find the docker we use in `benchmarking pipeline`
- Deploy the docker, and inside the docker:
  - Download `nightly-benchmarks.zip`.
  - In the same folder, run the following code:

  ```bash
  export HF_TOKEN=<your HF token>
  apt update
  apt install -y git
  unzip nightly-benchmarks.zip
  VLLM_SOURCE_CODE_LOC=./ bash .buildkite/nightly-benchmarks/scripts/run-nightly-benchmarks.sh
  ```

And the results will be inside `./benchmarks/results`.
