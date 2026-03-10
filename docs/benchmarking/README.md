# Benchmark Suites

vLLM provides comprehensive benchmarking tools for performance testing and evaluation:

- **[Benchmark CLI](./cli.md)**: `vllm bench` CLI tools and specialized benchmark scripts for interactive performance testing.
- **[Parameter Sweeps](./sweeps.md)**: Automate `vllm bench` runs for multiple configurations, useful for [optimization and tuning](../configuration/optimization.md).
- **[Performance Dashboard](./dashboard.md)**: Automated CI that publishes benchmarks on each commit.
## 测试指令模型的准确率（例如 GSM8K）

对于指令微调模型（如 Phi-4、Llama-3-Instruct 等），必须使用聊天补全接口才能获得符合预期的结果。以下示例展示如何对 GSM8K 测试集进行准确率评估。

### 启动服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/phi4-mini-instruct \
    --served-model-name phi4-mini \
    --port 8004 \
    --trust-remote-code

