# IndexCache

IndexCache reduces redundant top-k computation in DeepSeek-V3.2 (DSA) models by caching and reusing top-k indices across layers.

## Background

DeepSeek-V3.2 uses a DeepSeek Sparse Attention (DSA) mechanism where top-k token selection is computed per layer. For deep models with many layers, this computation can be expensive. IndexCache allows skipping redundant top-k computations by reusing indices from previous layers.

See: [IndexCache Paper](https://arxiv.org/abs/2603.12201)

## Usage

### CLI

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ...
```

### Configuration Reference

| Parameter            | Type | Default | Description                                                                                                                                      |
|----------------------|------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `use_index_cache`    | bool | false   | Enable IndexCache. Must be set to true to use this feature                                                                                       |
| `index_topk_freq`    | int  | 1       | Frequency (in layers) at which top-k is computed. 1 = compute on every layer (disabled), 4 = compute on 1/4 of layers                            |
| `index_topk_pattern` | str  | null    | Per-layer F/S pattern. Overrides index_topk_freq if set. Each character maps to one DSA layer: F = Full, S = Shared                              |

### Configuration Examples

**Using `index_topk_freq`** (compute every N layers):

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ...
```

**Using `index_topk_pattern`** (explicit per-layer control):

```bash
# custom pattern for 61 layers: F = compute, S = reuse
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSF"}'
```

## How It Works

1. When IndexCache is enabled, layers marked with `"F"` (Full) calculate and store top-k indices
2. Subsequent layers marked with `"S"` (Shared) receive the cached indices from the previous layer instead of recomputing
3. The cached indices are passed through the layer stack, reducing total computation

## Requirements

- DeepSeek-V3.2 or compatible DSA model
- `use_index_cache: true` via `--hf-overrides`
