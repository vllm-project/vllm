# LLM Compressor

LLM Compressor is a library for optimizing models for deployment with vLLM.
It provides a comprehensive set of quantization algorithms, including support for techniques such as FP8, INT8, and INT4 weight-only quantization.

## Why use LLM Compressor?

Modern LLMs often contain billions of parameters stored in 16-bit or 32-bit floating point, requiring substantial GPU memory and limiting deployment options.
Quantization lowers memory requirements while maintaining inference output quality by reducing the precision of model weights and activations to smaller data types.

LLM Compressor provides the following benefits:

- **Reduced memory footprint**: Run larger models on smaller GPUs.
- **Lower inference costs**: Serve more concurrent users per GPU, directly reducing the cost per query in production deployments.
- **Faster inference**: Smaller data types mean less memory bandwidth consumed, which often translates to higher throughput, especially for memory-bound workloads.

LLM Compressor handles the complexity of quantization, calibration, and format conversion, producing models ready for immediate use with vLLM.

## Key features

- **Multiple Quantization Methods**: Support for FP8, INT8, INT4, and mixed-precision quantization
- **One-Shot Quantization**: Quantize models quickly with minimal calibration data
- **vLLM Integration**: Seamlessly deploy quantized models with vLLM
- **Hugging Face Compatibility**: Works with models from the Hugging Face Hub

## Documentation

For the latest documentation, including installation guides, tutorials, and API reference, visit:

**[LLM Compressor Documentation](https://docs.vllm.ai/projects/llm-compressor/en/latest/)**

## Resources

- [LLM Compressor Documentation](https://docs.vllm.ai/projects/llm-compressor/en/latest/)
- [GitHub Repository](https://github.com/vllm-project/llm-compressor)
- [Quantization Guide](../features/quantization/README.md) - Using quantized models with vLLM
