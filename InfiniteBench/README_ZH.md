<div align="center">
  <img src="figs/InfiniteBench.jpg" width="500px"/>
  <br />
  <br />
  
# InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens

<p align="center">
  <a href="./README_ZH.md">中文</a> •
  <a href="./README.md">English</a> •
  <a href="https://arxiv.org/abs/2402.13718">论文</a>
</p>

</div>

## 简介

理解、处理长文本，是大模型迈向更深层次理解与交互阶段必备的能力。现已有大模型声称可以处理100k+的长序列，但是对应的标准评测集却是空缺的。为此，我们构建了一个面向 100k+ 的评测集，InfiniteBench。该评测集针对大模型在长文本方面的五项能力而设计：检索、数学、代码、问答、和摘要。

## 特点

- **长上下文:** InfiniteBench 测试数据的平均上下文长度为195k，远超现有评测数据。
- **多领域多语言:** InfiniteBench 评测集包含12个任务，包括中英双语，涵盖了检索、数学、代码、问答、和摘要等5个领域。
- **前瞻性挑战性:** InfiniteBench 测试任务，对标当前最强的模型如 GPT-4, Claude 2 等。
- **真实场景与合成场景:** InfiniteBench 既包含真实场景数据，探测大模型在处理实际问题的能力；也包含合成数据，为测试数据拓展上下文窗口提供了便捷。

## 任务构成

| Task Name            | Context       | # Examples | Avg Input Tokens | Avg Output Tokens | Description                                                                                 |
| -------------------- | ------------- | ---------- | ---------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| En.Sum               | Fake Book     | 103        | 171.5k           | 1.1k              | Summarization of a fake book created with core entity substitution.                         |
| En.QA                | Fake Book     | 351        | 192.6k           | 4.8               | Free-form question answering based on the fake book.                                        |
| En.MC                | Fake Book     | 229        | 184.4k           | 5.3               | Multiple choice questions derived from the fake book.                                       |
| En.Dia               | Script        | 200        | 103.6k           | 3.4               | Identification of talkers in partially anonymized scripts.                                  |
| Zh.QA                | New Book      | 175        | 2068.6k          | 6.3               | Question answering on a set of newly collected books.                                       |
| Code.Debug           | Code Document | 394        | 114.7k           | 4.8               | Finding which function in a code repo contains an crashing error (in multiple choice form). |
| Code.Run             | Synthetic     | 400        | 75.2k            | 1.3               | Simulating execution of multiple simple, synthetic functions.                               |
| Math.Calc            | Synthetic     | 50         | 43.9k            | 43.9k             | Calculations involving super-long arithmetic equations.                                     |
| Math.Find            | Synthetic     | 350        | 87.9k            | 1.3               | Finding special integers in a lengthy list.                                                 |
| Retrieve.PassKey[^1] | Synthetic     | 590        | 122.4k           | 2.0               | Retrieving hidden keys in a noisy long context.                                             |
| Retrieve.Number      | Synthetic     | 590        | 122.4k           | 4.0               | Locating repeated hidden numbers in a noisy long context.                                   |
| Retrieve.KV[^2]      | Synthetic     | 500        | 89.9k            | 22.7              | Finding the corresponding value from a dictionary and a key.                                |


## 评测结果

我们在 SOTA 模型上评测了 InfiniteBench 结果如下：

| Task Name        | GPT-4  | YaRN-Mistral-7B | Kimi-Chat | Claude 2 | Yi-6B-200K |  Yi-34B-200K |  Chatglm3-6B-128K |
| ---------------- | ------ | --------------- | --------- | -------- | -----------|  -----------| -----------|
| Retrieve.PassKey | 100%   | 92.71%          | 98.14%    | 97.80%   | 100.00%    | 100.00%    | 92.20%       |
| Retrieve.Number  | 100%   | 56.61%          | 95.42%    | 98.14%   | 94.92%     | 100.00%    | 80.68%      |
| Retrieve.KV      | 89.00% | < 5%            | 53.60%    | 65.40%   | < 5%       | < 5%       | < 5%       |
| En.Sum           | 14.73% | 9.09%           | 17.96%    | 14.50%   | < 5%       | < 5%       |< 5%       |
| En.QA            | 22.44% | 9.55%          | 16.52%    | 11.97%    |      9.20% |     12.17% |< 5%       |
| En.MC            | 67.25% | 27.95%          | 72.49%    | 62.88%   | 36.68%     |38.43%     |10.48%       |
| En.Dia           | 8.50%  | 7.50%           | 11.50%    | 46.50%   | < 5%       |< 5%       |< 5%       |
| Zh.QA            | 25.96% | 16.98%          | 17.93%    | 9.64%    | 15.07%     |13.61%       |< 5%       |
| Code.Debug       | 37.06% | < 5%            | 17.77%    | < 5%     | 9.14%       |13.96%       |7.36%       |
| Code.Run         | 23.25% | < 5%            | < 5%      | < 5%     | < 5%       |< 5%       |< 5%       |
| Math.Calc        | < 5%   | < 5%            | < 5%      | < 5%     | < 5%       |< 5%       |< 5%       |
| Math.Find        | 60.00% | 17.14%          | 12.57%    | 32.29%   | < 5%       |25.71%       |7.71%       |

注： 

1. YaRN-Mistral-7B 实现代码已开源在仓库，请大家批评指正；Kimi-Chat 和 Claude 2 使用用户界面评测，GPT-4 使用 API 评测，均使用官方默认配置。


## 评测

## 获取数据集

从 <https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench> 下载数据集到 `infinitebench/data` 路径下（我们将评测数据集放在 InfiniteBench 目录下），得到文件如下：

```
InfiniteBench
├── data
│   ├── code_debug.jsonl
│   ├── code_run.jsonl
│   ├── kv_retrieval.jsonl
│   ├── longbook_choice_eng.jsonl
│   ├── longbook_qa_chn.jsonl
│   ├── longbook_qa_eng.jsonl
│   ├── longbook_sum_eng.jsonl
│   ├── longdialogue_qa_eng.jsonl
│   ├── math_calc.jsonl
│   ├── math_find.jsonl
│   ├── number_string.jsonl
│   ├── passkey.jsonl
│   └── construct_synthetic_dataset.py
...
```

或者使用 Datasets 下载：

```python
from datasets import load_dataset, Value, Sequence
ft = Features({"id": Value("int64"), "context": Value("string"), "input": Value("string"), "answer": Sequence(Value("string")), "options": Sequence(Value("string"))})
dataset = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)
```

### 安装依赖

```shell
pip install -r requiremnets.txt
```

### 推理

比如，评测 GPT-4 在 Retrieve.PassKey 任务上的表现：

```shell
cd src
python eval_gpt4.py --task passkey
```

可以选择的 `--task` 有：

- `passkey`
- `number_string`
- `kv_retrieval`
- `longbook_sum_eng`
- `longbook_qa_eng`
- `longbook_qa_chn`
- `longbook_choice_eng`
- `longdialogue_qa_eng`
- `math_calc`
- `math_find`
- `code_debug`
- `code_run`

#### 计算分数

```shell
python compute_scores.py
```

## 引用

> This will be updated when our preprint paper is released.

```bibtex
@inproceedings{zhang-etal-2024-bench,
    title = "$\infty${B}ench: Extending Long Context Evaluation Beyond 100{K} Tokens",
    author = "Zhang, Xinrong  and
      Chen, Yingfa  and
      Hu, Shengding  and
      Xu, Zihang  and
      Chen, Junhao  and
      Hao, Moo  and
      Han, Xu  and
      Thai, Zhen  and
      Wang, Shuo  and
      Liu, Zhiyuan  and
      Sun, Maosong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.814",
    pages = "15262--15277",
    abstract = "Processing and reasoning over long contexts is crucial for many practical applications of Large Language Models (LLMs), such as document comprehension and agent construction. Despite recent strides in making LLMs process contexts with more than 100K tokens, there is currently a lack of a standardized benchmark to evaluate this long-context capability. Existing public benchmarks typically focus on contexts around 10K tokens, limiting the assessment and comparison of LLMs in processing longer contexts. In this paper, we propose , the first LLM benchmark featuring an average data length surpassing 100K tokens. comprises synthetic and realistic tasks spanning diverse domains in English and Chinese. The tasks in are designed to require an understanding of long dependencies in contexts and make simply retrieving a limited number of passages from contexts not sufficient for these tasks. Based on , we evaluate several state-of-the-art LLMs tailored for processing long contexts. The experimental results indicate that existing long-context LLMs still require significant advancements to process 100K+ contexts effectively. Furthermore, we present three intriguing analyses regarding the behavior of LLMs processing long context. Our code and data is released.",
}
```

## 参考文献
[^1]: Mohtashami, Amirkeivan and Martin Jaggi. “Landmark Attention: Random-Access Infinite Context Length for Transformers.” ArXiv abs/2305.16300 (2023): n. pag.
[^2]: Liu, Nelson F. et al. “Lost in the Middle: How Language Models Use Long Contexts.” ArXiv abs/2307.03172 (2023): n. pag.
