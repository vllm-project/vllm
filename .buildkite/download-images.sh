#!/bin/bash

set -e

# Missing AWS credentials
# mkdir -p ../tests/images/
# aws s3 sync s3://air-example-data-2/vllm_opensource_llava/ ../tests/images/
cd ../tests
mkdir -p images
cd images
wget https://air-example-data-2.s3.us-west-2.amazonaws.com/vllm_opensource_llava/stop_sign_pixel_values.pt
wget https://air-example-data-2.s3.us-west-2.amazonaws.com/vllm_opensource_llava/stop_sign_image_features.pt
wget https://air-example-data-2.s3.us-west-2.amazonaws.com/vllm_opensource_llava/cherry_blossom_pixel_values.pt
wget https://air-example-data-2.s3.us-west-2.amazonaws.com/vllm_opensource_llava/cherry_blossom_image_features.pt
