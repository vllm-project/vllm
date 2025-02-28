#!/bin/bash

echo "vLLM linting system has been moved from format.sh to pre-commit hook."
echo "Please run 'pip install -r requirements-lint.txt', followed by"
echo "'pre-commit install --hook-type pre-commit --hook-type commit-msg' to install the pre-commit hook."
echo "Then linters will run automatically before each commit."
