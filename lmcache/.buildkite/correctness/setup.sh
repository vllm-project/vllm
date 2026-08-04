#!/bin/bash

# buildkite agents start at the root of the Git checkout

cd .buildkite/correctness

unxz data.tar.xz
tar xf data.tar

# see README.md for pre-configuring your CI runner
source ~/correctness_venv/bin/activate
cd ~/correctness_repositories/LMCache
git pull origin dev
cd ~/correctness_repositories/vllm
git pull origin main
