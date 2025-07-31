# Long Text Embedding with Chunked Processing

This directory contains examples for using vLLM's **chunked processing** feature to handle long text embedding that exceeds the model's maximum context length.

## 🚀 Quick Start

### 1. Start the Server

Use the provided script to start a vLLM server with chunked processing enabled:

```bash
# Basic usage (supports very long texts up to ~3M tokens)
./openai_embedding_long_text_service.sh

# Custom configuration with different models
MODEL_NAME="jinaai/jina-embeddings-v3" \
MAX_EMBED_LEN=1048576 \
./openai_embedding_long_text_service.sh

# For extremely long documents
MODEL_NAME="intfloat/multilingual-e5-large" \
MAX_EMBED_LEN=3072000 \
./openai_embedding_long_text_service.sh
```

### 2. Test Long Text Embedding

Run the comprehensive test client:

```bash
python openai_embedding_long_text_client.py
```

## 📁 Files

| File | Description |
|------|-------------|
| `openai_embedding_long_text_service.sh` | Server startup script with chunked processing enabled |
| `openai_embedding_long_text_client.py` | Comprehensive test client for long text embedding |
| `openai_embedding_client.py` | Basic embedding client (updated with chunked processing info) |

## ⚙️ Configuration

### Server Configuration

The key parameters for chunked processing are in the `--override-pooler-config`:

```json
{
  "pooling_type": "MEAN",
  "normalize": true,
  "enable_chunked_processing": true,
  "max_embed_len": 3072000
}
```

#### Chunked Processing Behavior

Chunked processing now uses **MEAN aggregation** for cross-chunk combination, regardless of the model's native pooling type:

| Component | Behavior | Description |
|-----------|----------|-------------|
| **Within chunks** | Native pooling (MEAN/CLS/LAST) | Uses model's original pooling strategy |
| **Cross-chunk aggregation** | Always MEAN | Weighted averaging based on chunk token counts |
| **Performance** | Optimal | All chunks processed for complete semantic coverage |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `intfloat/multilingual-e5-large` | Embedding model to use (supports multiple models) |
| `PORT` | `31090` | Server port |
| `GPU_COUNT` | `1` | Number of GPUs to use |
| `MAX_EMBED_LEN` | `3072000` | Maximum embedding input length (supports very long documents) |
| `POOLING_TYPE` | `auto` | Model's native pooling type: `auto`, `MEAN`, `CLS`, `LAST` |
| `API_KEY` | `EMPTY` | API key for authentication |

## 🔧 How It Works

1. **Enhanced Input Validation**: `max_embed_len` allows accepting inputs longer than `max_model_len` without environment variables
2. **Smart Chunking**: Text is split based on `max_position_embeddings` to maintain semantic integrity
3. **Unified Processing**: All chunks processed separately through the model using native pooling
4. **MEAN Aggregation**: Results combined using token count-based weighted averaging across all chunks
5. **Consistent Output**: Final embeddings maintain the same dimensionality as standard processing

### Input Length Handling

- **Within max_embed_len**: Input is accepted and processed (up to 3M+ tokens)
- **Exceeds max_position_embeddings**: Chunked processing is automatically triggered
- **Exceeds max_embed_len**: Input is rejected with clear error message
- **No environment variables required**: Works without `VLLM_ALLOW_LONG_MAX_MODEL_LEN`

### Extreme Long Text Support

With `MAX_EMBED_LEN=3072000`, you can process:
- **Academic papers**: Full research papers with references
- **Legal documents**: Complete contracts and legal texts  
- **Books**: Entire chapters or small books
- **Code repositories**: Large codebases and documentation

## 📊 Performance Characteristics

### Chunked Processing Performance

| Aspect | Behavior | Performance |
|--------|----------|-------------|
| **Chunk Processing** | All chunks processed with native pooling | Consistent with input length |
| **Cross-chunk Aggregation** | MEAN weighted averaging | Minimal overhead |
| **Memory Usage** | Proportional to number of chunks | Moderate, scalable |
| **Semantic Quality** | Complete text coverage | Optimal for long documents |

## 🧪 Test Cases

The test client demonstrates:

- ✅ **Short text**: Normal processing (baseline)
- ✅ **Medium text**: Single chunk processing
- ✅ **Long text**: Multi-chunk processing with aggregation
- ✅ **Very long text**: Many chunks processing
- ✅ **Extreme long text**: Document-level processing (100K+ tokens)
- ✅ **Batch processing**: Mixed-length inputs in one request
- ✅ **Consistency**: Reproducible results across runs

## 🐛 Troubleshooting

### Common Issues

1. **Chunked processing not enabled**:

   ```
   ValueError: This model's maximum position embeddings length is 4096 tokens...
   ```

   **Solution**: Ensure `enable_chunked_processing: true` in pooler config

2. **Input exceeds max_embed_len**:

   ```
   ValueError: This model's maximum embedding input length is 3072000 tokens...
   ```

   **Solution**: Increase `max_embed_len` in pooler config or reduce input length

3. **Memory errors**:
  
   ```
   RuntimeError: CUDA out of memory
   ```
  
   **Solution**: Reduce chunk size by adjusting model's `max_position_embeddings` or use fewer GPUs

4. **Slow processing**:
   **Expected**: Long text takes more time due to multiple inference calls

### Debug Information

Server logs show chunked processing activity:

```
INFO: Input length 150000 exceeds max_position_embeddings 4096, will use chunked processing
INFO: Split input of 150000 tokens into 37 chunks (max_chunk_size: 4096)
```

## 📚 Additional Resources

- [Pooling Models Documentation](../../docs/models/pooling_models.md#chunked-processing-for-long-text)
- [Supported Models List](../../docs/models/supported_models.md#text-embedding)
- [Original Feature Documentation](../../README_CHUNKED_PROCESSING.md)

## 🤝 Contributing

To extend chunked processing support to other embedding models:

1. Check model compatibility with the pooling architecture
2. Test with various text lengths
3. Validate embedding quality compared to single-chunk processing
4. Submit PR with test cases and documentation updates

## 🆕 Enhanced Features

### max_embed_len Parameter

The new `max_embed_len` parameter provides:

- **Simplified Configuration**: No need for `VLLM_ALLOW_LONG_MAX_MODEL_LEN` environment variable
- **Flexible Input Validation**: Accept inputs longer than `max_model_len` up to `max_embed_len`
- **Extreme Length Support**: Process documents with millions of tokens
- **Clear Error Messages**: Better feedback when inputs exceed limits
- **Backward Compatibility**: Existing configurations continue to work

---

**Note**: Chunked processing is currently supported for specific embedding models. See the [supported models documentation](../../docs/models/supported_models.md#chunked-processing-for-long-text) for the complete list.
