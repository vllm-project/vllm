# 大规模推测解码吞吐评测

这套工具用于比较不同推测解码方法的在线服务吞吐，首要指标是输出吞吐
`output tokens/s`。

评测矩阵包括：

- 模型：HY3-295B-A21B、Qwen3-8B；
- HY3 方法：AR、MTP-3、DFlash-7、DFly-7、DFly-7 + D-Cut；
- Qwen3-8B 方法：AR、MTP-3、DFlash-7、DSpark-7、DFly-7、
  DFly-7 + D-Cut；
- 数据集：MATH-500、HumanEval、GSM8K、MT-Bench、LiveCodeBench、MBPP；
- 闭环并发：4、8、16、32、64、128。

每个矩阵单元先预热 30 秒，再执行 3 个 120 秒测量窗口。结果记录输出
吞吐、请求吞吐、总 token 吞吐、平均接受长度（MAT）和 draft token
接受率。MAT 是辅助诊断指标，不是本工具的主要评测目标。

这是性能评测工具，不负责准确率打分。处理后的每行 JSONL 必须包含一个
`prompt`，并作为独立的单轮请求发送。原始 `load_from_disk` 数据集、
MT-Bench 多轮对话和 judge 打分不属于本评测协议。

## 默认数据

- 处理后数据集：
  `/apdcephfs_sgfd2/share_300532381/ruicen/draft_models/processed_jsonl_datasets`

## 推理与采样设置

完整吞吐矩阵统一使用以下请求设置：

- API：`/v1/chat/completions`
- 思考模式：关闭
    - `reasoning_effort="no_think"`，用于 HY3
    - `enable_thinking=false`，用于 Qwen3
    - 每个模型只传递自己对应的 `chat_template_kwargs`
- `temperature=0.0`
- `top_k=1`
- `top_p=1.0`
- `seed=0`
- `max_completion_tokens=4096`
- `n=1`，不设置额外 stop token
- 自然 EOS，`ignore_eos=false`
- 流式返回，`stream=true`，并开启 usage 统计
- 不额外设置 repetition、presence 或 frequency penalty，使用 vLLM 默认值

统一的 server 设置为：

- `max_model_len=8192`
- `max_num_batched_tokens=16384`
- `max_num_seqs=128`
- `gpu_memory_utilization=0.97`
- HY3 使用 TP8，Qwen3-8B 使用 TP1
- 非 D-Cut 方法使用 `FULL_AND_PIECEWISE` CUDA graph
- DFly + D-Cut 使用 `PIECEWISE` CUDA graph
- 禁用 prefix caching，避免循环使用评测 prompt 时缓存命中污染吞吐
- 禁用 async scheduling，与参考吞吐评测协议保持一致
- `VLLM_USE_V2_MODEL_RUNNER=0`
- `VLLM_DSPARK_HPC_CORRECTION=1`

除非要单独研究 thinking 模式，否则完整矩阵不要修改这些参数。请求设置会
写入 `protocol.json` 和单元结果。

## 使用方式

SSH 机器已准备好运行环境。进入机器上的代码目录后直接执行：

```bash
cd <代码目录>
```

无需 GPU，先检查解析后的全部 server 命令：

```bash
.venv/bin/python benchmarks/spec_decode/run_evaluation.py \
  --model-family hy3 \
  --dry-run
```

大规模运行时，建议每台机器只测试一种方法：

```bash
.venv/bin/python benchmarks/spec_decode/run_evaluation.py \
  --model-family hy3 \
  --methods dfly-7-dcut \
  --output-dir /path/to/results/hy3-dfly-7-dcut
```

提交完整矩阵前，先执行一个短 smoke：

```bash
.venv/bin/python benchmarks/spec_decode/run_evaluation.py \
  --model-family qwen3-8b \
  --methods dfly-7-dcut \
  --datasets gsm8k \
  --concurrencies 4 \
  --warmup-seconds 2 \
  --measure-seconds 10 \
  --repeats 1 \
  --max-prompts 8 \
  --output-dir /path/to/results/qwen3-smoke
```

默认启用断点续跑，同一输出目录中的已完成单元会自动跳过。更换 checkpoint
时应使用新输出目录，或通过 `--no-resume` 强制重跑。runner 只停止自己
启动的 server 进程组，不使用 `pkill`。

## 模型配置

模型和 speculative config 集中在 `configs.json`，需要换 checkpoint 时只
修改这个文件。模型路径已统一到 `sgfd2`，Qwen3 MTP 使用已经转换为
Hugging Face 格式的 `iter_0013831_served`。

## 输出

输出目录包括：

- `protocol.json`：模型、方法、请求设置和评测参数；
- `logs/server_*.log`：各方法的 server 日志；
- `logs/cell_*.log`：各矩阵单元的客户端日志；
- `cells/<method>/<dataset>_c<concurrency>.json`：完整窗口指标；
- `summary.json`：所有已完成单元的吞吐和接受率摘要。
