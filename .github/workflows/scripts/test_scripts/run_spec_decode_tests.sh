#!/usr/bin/env bash
# Run speculative decoding tests.
# Requires aws.g5.2xlarge or larger.

set -eou pipefail

timeout $(( 5 * 60 )) pytest -vvs tests/samplers/test_rejection_sampler.py
