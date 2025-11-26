# KV Cache Cartridge Usage Guide

This guide explains how to use KV cache cartridges with vLLM to load pre-computed KV caches from S3 or local storage.

## Overview

KV cache cartridges allow you to load pre-computed KV cache data with your inference requests. This can significantly speed up inference when you have common prompt prefixes that can be pre-computed and cached.

## Cartridge Format

Cartridges are saved as PyTorch `.pt` files with the following structure:

```python
{
    'token_ids': torch.Tensor or list[int],  # The token IDs this cache represents
    'kv_cache': dict[str, torch.Tensor],     # Optional: Per-layer KV cache tensors
    'metadata': dict,                         # Optional: Additional metadata
}
```

**Minimal format:**
```python
{
    'token_ids': [101, 2023, 2003, ...],  # List of token IDs
}
```

## API Usage

### Chat Completion Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "cartridges": [
            {
                "id": "s3://my-bucket/path/to/cartridge.pt",
                "source": "s3",
                "force_redownload": False
            }
        ]
    }
)
```

### Completion Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "cartridges": [
            {
                "id": "/path/to/local/cartridge.pt",
                "source": "local",
                "force_redownload": False
            }
        ]
    }
)
```

## Cartridge Parameters

### `id` (required)
- For S3: `s3://bucket-name/path/to/cartridge.pt`
- For local: `/absolute/path/to/cartridge.pt`

### `source` (optional, default: "s3")
- `"s3"`: Load from Amazon S3
- `"local"`: Load from local filesystem

### `force_redownload` (optional, default: false)
- `true`: Force re-download even if cached
- `false`: Use cached version if available

## How It Works

1. **Request Processing**: When a request includes cartridges, vLLM loads them before processing
2. **Token Prepending**: Cartridge token IDs are prepended to your prompt tokens
3. **Prefix Caching**: The combined tokens benefit from vLLM's built-in prefix caching
4. **Cache Reuse**: Identical cartridge prefixes are automatically reused across requests

## S3 Configuration

To use S3 cartridges, ensure:

1. **Install boto3**: `pip install boto3`
2. **Configure AWS credentials**:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

Or use IAM roles if running on EC2/ECS.

## Cache Directory

Downloaded cartridges are cached locally in:
```
~/.cache/vllm/cartridges/
```

This prevents re-downloading cartridges on every request.

## Creating Cartridges

Example script to create a cartridge:

```python
import torch

# Tokenize your common prefix
token_ids = [101, 2023, 2003, 1037, 2400, ...]  # Example token IDs

# Save as cartridge
cartridge = {
    'token_ids': token_ids,
    'metadata': {
        'model': 'meta-llama/Llama-3.1-8B',
        'description': 'Common system prompt',
    }
}

torch.save(cartridge, 'my_cartridge.pt')
```

Then upload to S3:
```bash
aws s3 cp my_cartridge.pt s3://my-bucket/cartridges/my_cartridge.pt
```

## Multiple Cartridges

You can specify multiple cartridges - they will be loaded in order:

```json
{
    "cartridges": [
        {
            "id": "s3://my-bucket/system-prompt.pt",
            "source": "s3"
        },
        {
            "id": "s3://my-bucket/context.pt",
            "source": "s3"
        }
    ]
}
```

## Best Practices

1. **Use for Static Prefixes**: Cartridges work best for prompts that don't change
2. **Reasonable Size**: Keep cartridges under a few thousand tokens for optimal performance
3. **Monitor Cache**: The cache directory can grow large with many unique cartridges
4. **Error Handling**: Handle cartridge loading errors gracefully in production

## Troubleshooting

### "boto3 is required" Error
```bash
pip install boto3
```

### "Failed to download from S3" Error
- Check AWS credentials
- Verify S3 bucket permissions
- Ensure bucket and key exist

### "Invalid cartridge format" Error
- Ensure `.pt` file contains required `token_ids` field
- Verify token IDs are valid for your model's tokenizer
