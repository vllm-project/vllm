<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

For GPU-accelerated inference on Apple Silicon, use [vLLM-Metal](https://github.com/vllm-project/vllm-metal), a community-maintained hardware plugin that uses MLX as the compute backend and provides native GPU acceleration via Apple's Metal framework.

vLLM-Metal works with MLX-optimized models from the [mlx-community](https://huggingface.co/mlx-community) organization on Hugging Face, which provides quantized versions of popular models optimized for Apple Silicon.

!!! tip
    For installation and usage instructions, see the [Set up using vLLM-Metal](#set-up-using-vllm-metal) section below.

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: macOS Sonoma or later
- Hardware: Apple Silicon
- Metal support enabled

!!! note
    See the [Set up using vLLM-Metal](#set-up-using-vllm-metal) section below for installation instructions.

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

## Set up using vLLM-Metal

vLLM-Metal is distributed as a separate package that provides native GPU acceleration on Apple Silicon.

To install vLLM-Metal, follow the installation instructions in the [vLLM-Metal documentation](https://github.com/vllm-project/vllm-metal#installation).

The installation will:

1. Set up the appropriate Python environment
2. Install MLX and required dependencies
3. Install the vLLM-Metal package

After installation, you can start using vLLM with Metal GPU acceleration.

!!! tip
    When using vLLM-Metal, use models from the [mlx-community](https://huggingface.co/mlx-community) on Hugging Face for best performance. These models are optimized for MLX and often include quantized versions (4-bit, 8-bit) that run efficiently on Apple Silicon.

    Example model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

### Using vLLM-Metal

After installation, vLLM-Metal provides an easy-to-use CLI for running an OpenAI-compatible API server:

```bash
# Activate the vLLM-Metal environment
source ~/.venv-vllm-metal/bin/activate

# Start the API server (specify your mlx-community model or it will use default)
vllm serve
```

Once the server is running, you have multiple options to interact with it:

#### Option 1: Interactive chat

Open a new terminal and start an interactive chat session:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm chat
```

#### Option 2: API requests with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

#### Option 3: Python with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # No auth required for local server
)

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

For more details on the `vllm` CLI commands, see the [OpenAI-compatible server documentation](../../serving/openai_compatible_server.md).

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

vLLM-Metal is installed via the vLLM-Metal package. See the [Set up using vLLM-Metal](#set-up-using-vllm-metal) section above.

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

For build instructions from source, refer to the [vLLM-Metal documentation](https://github.com/vllm-project/vllm-metal#installation).

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

vLLM-Metal provides:

- Native GPU acceleration using Metal
- MLX-based compute backend optimized for Apple Silicon
- OpenAI-compatible API server
- Support for popular model architectures

For specific feature support and limitations, refer to the [vLLM-Metal documentation](https://github.com/vllm-project/vllm-metal).

--8<-- [end:supported-features]
