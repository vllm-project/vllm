# Long Text Embedding with Chunked Processing

This directory contains examples for using vLLM's **chunked processing** feature to handle long text embedding that exceeds the model's maximum context length.

## 🚀 Quick Start

### 1. Start the Server

Use the provided script to start a vLLM server with chunked processing enabled:

```bash
# Basic usage
./openai_embedding_long_text_service.sh

# Custom configuration
MODEL_NAME="intfloat/multilingual-e5-large" \
PORT=31090 \
MAX_MODEL_LEN=10240 \
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

The key parameter for chunked processing is in the `--override-pooler-config`:

```json
{
  "pooling_type": "CLS",
  "normalize": true,
  "enable_chunked_processing": true
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `intfloat/multilingual-e5-large` | Embedding model to use |
| `PORT` | `31090` | Server port |
| `GPU_COUNT` | `1` | Number of GPUs to use |
| `MAX_MODEL_LEN` | `10240` | Maximum model context length |
| `API_KEY` | `EMPTY` | API key for authentication |

## 🔧 How It Works

1. **Automatic Detection**: When input text exceeds `max_model_len`, chunked processing is triggered
2. **Smart Chunking**: Text is split at token boundaries to maintain semantic integrity
3. **Independent Processing**: Each chunk is processed separately through the model
4. **Weighted Aggregation**: Results are combined using token count-based weighted averaging
5. **Consistent Output**: Final embeddings maintain the same dimensionality as standard processing

## 📊 Performance Characteristics

| Text Length | Processing Method | Memory Usage | Speed |
|-------------|------------------|--------------|-------|
| ≤ max_len | Standard | Normal | Fast |
| > max_len | Chunked | Reduced per chunk | Slower (multiple inferences) |

## 🧪 Test Cases

The test client demonstrates:

- ✅ **Short text**: Normal processing (baseline)
- ✅ **Medium text**: Single chunk processing
- ✅ **Long text**: Multi-chunk processing with aggregation
- ✅ **Very long text**: Many chunks processing
- ✅ **Batch processing**: Mixed-length inputs in one request
- ✅ **Consistency**: Reproducible results across runs

## 🐛 Troubleshooting

### Common Issues

1. **Chunked processing not enabled**:

   ```
   ValueError: This model's maximum context length is 512 tokens...
   ```

   **Solution**: Ensure `enable_chunked_processing: true` in pooler config

2. **Memory errors**:
  
```
   RuntimeError: CUDA out of memory
   ```
  
**Solution**: Reduce `MAX_MODEL_LEN` or use fewer GPUs

1. **Slow processing**:
   **Expected**: Long text takes more time due to multiple inference calls

### Debug Information

Server logs show chunked processing activity:

```
INFO: Input length 15000 exceeds max_model_len 10240, will use chunked processing
INFO: Split input of 15000 tokens into 2 chunks
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

---

**Note**: Chunked processing is currently supported for specific embedding models. See the [supported models documentation](../../docs/models/supported_models.md#chunked-processing-for-long-text) for the complete list.
