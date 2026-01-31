#!/bin/bash

# Define the prefix for environment variables to look for
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker, https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
ARGS=(--port 8080)

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase, and replace underscores with dashes
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    # Add the argument name and value to the ARGS array
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Pass the collected arguments to the main entrypoint
exec standard-supervisor vllm serve "${ARGS[@]}"