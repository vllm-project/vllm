# Baseline accuracies for models of interest

## LLMs (lm_eval on gsm8k)

### DeepSeek-R1 Block-Scale FP8

deepseek-ai/DeepSeek-R1
```shell
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9492|±  |0.0060|
|     |       |strict-match    |     5|exact_match|↑  |0.9484|±  |0.0061|
```

### DeepSeek-R1 PTPC FP8

EmbeddedLLM/deepseek-r1-FP8-Dynamic
```shell
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9477|±  |0.0061|
|     |       |strict-match    |     5|exact_match|↑  |0.9469|±  |0.0062|
```

### Qwen3-Coder PTPC Quark FP8

EmbeddedLLM/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic
```shell
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|_  |0.8848|_  |0.0088|
|     |       |strict-match    |     5|exact_match|_  |0.8590|_  |0.0096|
```

### Qwen3-Next

Qwen/Qwen3-Next-80B-A3B-Instruct
```shell
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|_  |0.8537|_  |0.0097|
|     |       |strict-match    |     5|exact_match|_  |0.8135|_  |0.0107|

```

## VLMs and Omni Models (mistral-eval on chartqa)

### Qwen2.5-VL-72B
Qwen/Qwen2.5-VL-72B-Instruct
```shell
Metrics:
{
    "explicit_prompt_relaxed_correctness": 0.8652,
    "anywhere_in_answer_relaxed_correctness": 0.8828
}
```

### Qwen2.5-VL-72B PTPC FP8
RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic
```shell
Metrics:
{
    "explicit_prompt_relaxed_correctness": 0.8792,
    "anywhere_in_answer_relaxed_correctness": 0.8888
}
```

### Qwen3-VL-235B
Qwen/Qwen3-VL-235B-A22B-Instruct
```shell
Metrics:
{
    "explicit_prompt_relaxed_correctness": 0.8736,
    "anywhere_in_answer_relaxed_correctness": 0.8752
}
```

### Qwen3-VL-235B PTPC FP8
RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
```shell
Metrics:
{
    "explicit_prompt_relaxed_correctness": 0.8724,
    "anywhere_in_answer_relaxed_correctness": 0.874
}
```

### Qwen3-Omni
Qwen/Qwen3-Omni-30B-A3B-Instruct
```shell
Metrics:
{
    "explicit_prompt_relaxed_correctness": 0.8736,
    "anywhere_in_answer_relaxed_correctness": 0.8768
}
```
