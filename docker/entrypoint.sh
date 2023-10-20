#!/bin/bash

if [[ ! -z "${HF_TOKEN}" ]]; then
    echo "The HF_TOKEN environment variable set, logging to Hugging Face."
    python3 -c "import huggingface_hub; huggingface_hub.login('${HF_TOKEN}')"
else
    echo "The HF_TOKEN environment variable is not set or empty, not logging to Hugging Face."
fi

# Run the provided command
exec python3 tgi/api_server.py "$@"