# Disaggregated Encoder

These example scripts that demonstrate the disaggregated encoder (EPD) features of vLLM.

For a detailed explanation of the EPD features, please refer to the [Disaggregated Encoder Feature Documentation](../../../docs/features/disagg_encoder.md).

## Files

- `disagg_epd_proxy.py` - Proxy script that demonstrates the XeYpZd setup (X encode instances, Y prefill instances, Z decode instances). Currently stable for the 1e1p1d configuration.

- `disagg_1e1p1d_example.sh` - Sets up the 1e1p1d configuration, runs the VisionArena benchmark, and processes a single request with a local image.

- `disagg_1e1pd_example.sh` - Sets up the 1e1pd configuration, runs the VisionArena benchmark, and processes a single request with a local image.

### Custom Configuration

```bash
# Use specific GPUs
GPU_E=0 GPU_PD=1 GPU_P=1 GPU_D=2 bash disagg_1e1p1d_example.sh

# Use specific ports
ENDPOINT_PORT=10001 bash disagg_1e1p1d_example.sh

# Use specific model
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" bash disagg_1e1p1d_example.sh

# Use specific storage path
EC_SHARED_STORAGE_PATH="/tmp/my_ec_cache" bash disagg_1e1p1d_example.sh
```

## Encoder Instances

Encoder engines should be launched with the following flags:

- `--enforce-eager` **(required)** – The current EPD implementation is only compatible with encoder instances running in this mode.

- `--no-enable-prefix-caching` **(required)** – Encoder instances do not consume KV cache; prefix caching is disabled to avoid conflicts with other features.

- `--max-num-batched-tokens=<large value>` **(default: 2048)** – This flag controls the token scheduling budget per decoding step and is irrelevant to encoder-only instances. **Set it to a very high value (effectively unlimited) to bypass scheduler limitations.** The actual token budget is managed by the encoder cache manager.

- `--mm-encoder-only` **(Optional)** - The language model is skipped during initialization to reduce device memory usage. **Models using this option must initialize the language component inside the context of `SupportsMultiModal._mark_language_model`.**

## Local media inputs

To support local image inputs (from your ```MEDIA_PATH``` directory), add the following flag to the encoder instance:

```bash
--allowed-local-media-path $MEDIA_PATH
```

The vllm instances and `disagg_encoder_proxy` supports local URIs with ```{"url": "file://'"$MEDIA_PATH_FILENAME"'}``` as multimodal inputs. Each URI is passed unchanged from the `disagg_encoder_proxy` to the encoder instance so that the encoder can load the media locally.

## EC connector and KV transfer

The `ECExampleonnector` is used to store the encoder cache on local disk and facilitate transfer. To enable the encoder disaggregation feature, add the following configuration:

```bash
# Add to encoder instance: 
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_producer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
    }
}' 

# Add to prefill/prefill+decode instance: 
--ec-transfer-config '{
    "ec_connector": "ECExampleConnector",
    "ec_role": "ec_consumer",
    "ec_connector_extra_config": {
        "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
    }
}' 
```

`$EC_SHARED_STORAGE_PATH` is the path where the EC connector temporarily stores the cache.

If you enable prefill instance (`--prefill-servers-urls` not disabled), you will need --kv-transfer-config to facilitate the PD disaggregation. Currently, we use the `NixlConnector` for this purpose. Refer to `tests/v1/kv_connector/nixl_integration` for more example codes on PD disaggregation with Nixl.

```bash
# Add to prefill instance:    
--kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer"
}' 

# Add to decode instance:
--kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer"
}' 
```

## Proxy Instance Flags (`disagg_epd_proxy.py`)

| Flag | Description |
|------|-------------|
| `--encode-servers-urls` | Comma-separated list of encoder endpoints. Every multimodal item extracted from the request is fanned out to one of these URLs in a round-robin fashion. |
| `--prefill-servers-urls` | Comma-separated list of prefill endpoints. Set to `disable`, `none`, or `""` to skip the dedicated prefill phase and run E+PD (encoder + combined prefill/decode). |
| `--decode-servers-urls` | Comma-separated list of decode endpoints. Non-stream and stream paths both round-robin over this list. |
| `--host`, `--port` | Bind address for the proxy itself (defaults: `0.0.0.0:8000`). |

Example usage:
For E + PD setup:

```bash
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8002" \
      --prefill-servers-urls "disable" \
      --decode-servers-urls "http://pd1:8003,http://pd2:8004"
```

For E + P + D setup:

```bash
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8001" \
      --prefill-servers-urls "http://p1:8003,http://p2:8004" \ 
      --decode-servers-urls "http://d1:8005,http://d2:8006"
```
