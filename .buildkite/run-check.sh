#!/bin/bash
set -euo pipefail

SECRET=$(aws secretsmanager get-secret-value --secret-id "hf_token" --query SecretString --output text)
echo "HI"
echo $SECRET