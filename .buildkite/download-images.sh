#!/bin/bash

set -e
mkdir -p images/
aws s3 sync s3://air-example-data-2/vllm_opensource_llava/ images/
