#!/bin/bash

# ==============================================================================
# vLLM macOS Apple Silicon (M1/M2/M3) Build & Run Reference Script
# ==============================================================================

# 1. Clear conflicting compiler flags
# Older tutorials might suggest setting CMAKE_ARGS for __builtin_clzg, 
# but this breaks compilation on newer Apple Clang (Xcode 16+).
unset CMAKE_ARGS
unset CXXFLAGS

# 2. Set the correct SDK path
# Ensure the SDK used by the compiler matches your Xcode version to avoid 
# undeclared identifier '__builtin_ctzg' errors in the C++ <bit> header.
# You can use `xcrun --show-sdk-path` or specify the SDK explicitly:
export SDKROOT=$(xcrun --show-sdk-path)
# If the above fails, uncomment and adjust the line below to match your SDK:
# export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk

# 3. Install dependencies
echo "Installing CPU requirements..."
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match

# 4. Compile and install vLLM from source
echo "Compiling and installing vLLM (this may take a few minutes)..."
uv pip install -e .

echo "Installation complete!"
echo ""
echo "=============================================================================="
echo "Usage Examples:"
echo "=============================================================================="
echo ""
echo "1. Run vLLM as an OpenAI-compatible API server:"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "       --model Qwen/Qwen2.5-0.5B-Instruct \\"
echo "       --gpu-memory-utilization 0.4 \\"
echo "       --port 8000 \\"
echo "       --trust-remote-code"
echo ""
echo "2. Query the server (in a new terminal):"
echo "   curl http://localhost:8000/v1/chat/completions \\"
echo "       -H \"Content-Type: application/json\" \\"
echo "       -d '{"
echo "           \"model\": \"Qwen/Qwen2.5-0.5B-Instruct\","
echo "           \"messages\": ["
echo "               {\"role\": \"user\", \"content\": \"你好，请用一句话介绍你自己。\"}"
echo "           ]"
echo "       }'"
echo ""
echo "3. Or verify using the Python script:"
echo "   python verify_cpu.py"
