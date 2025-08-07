"""
Example script for serving GPT-OSS model with vLLM OpenAI-compatible API server.

This script demonstrates how to serve the GPT-OSS model with reasoning capabilities
and tool server integration.

Usage:
    python openai_response_api_gpt_oss.py

The script will start a vLLM server with GPT-OSS model and demo tools.
You can then make requests to the server using the OpenAI API format.

Example request with reasoning:
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "openai/gpt-oss-120b",
        "messages": [
          {"role": "user", "content": "Solve this math problem: 2 + 2 = ?"}
        ],
        "include_reasoning": true,
        "temperature": 0.1
      }'

Example request with tools:
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "openai/gpt-oss-120b", 
        "messages": [
          {"role": "user", "content": "Calculate 15 * 23 using the calculator tool"}
        ],
        "tools": [
          {
            "type": "function",
            "function": {
              "name": "calculator",
              "description": "Perform basic calculations",
              "parameters": {
                "type": "object",
                "properties": {
                  "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                  }
                },
                "required": ["expression"]
              }
            }
          }
        ]
      }'
"""

import asyncio
import subprocess
import sys
from pathlib import Path


def main():
    """Start the vLLM server with GPT-OSS model configuration."""

    # Check if running in appropriate environment
    try:
        import vllm

        print(f"Using vLLM version: {vllm.__version__}")
    except ImportError:
        print("Error: vLLM not installed. Please install vLLM first.")
        sys.exit(1)

    # Command to start vLLM server with GPT-OSS
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "openai/gpt-oss-120b",  # or gpt-oss-20b for smaller model
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",  # Adjust based on your GPU setup
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-batched-tokens",
        "1024",  # Reduce if you encounter OOM
        "--tool-server",
        "demo",  # Enable demo tool server
        "--enable-auto-tool-choice",
        "--served-model-name",
        "gpt-oss",
        # Uncomment below for better performance on H100/B200
        # "--kv-cache-dtype", "fp8",
        # "--quantization", "mxfp4",  # Enable MXFP4 quantization
    ]

    print("Starting vLLM server with GPT-OSS model...")
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")

    try:
        # Run the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: vLLM not found. Please ensure vLLM is properly installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
