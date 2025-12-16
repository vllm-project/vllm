# 发布vLLM

vLLM的发行版提供了代码库的稳定版本，并以二进制格式打包，用户可以通过PyPI获取。这些发行版也作为开发团队的里程碑，向社区传达新功能，改进以及未来可能影响用户的更新，也包括潜在的破坏性变更。

## 版本发布

vLLM采用一种“右移式”的版本策略，每两周发布一次新的补丁。补丁内容包括新功能和bug修复（与语义化版本中补丁版本仅包含向后兼容的 bug 修复不同）。当需要做出重要修复时，会发布一个特殊的post1版本。

* 主版本（_major_） 在有重大架构里程碑时发布，或者像PyTorch 2.0那样引入不兼容的API更改时发布。
* 次版本（_minor_） 在有新功能或特性时发布。
* 补丁版本（_patch_） 在有新特性和向后兼容bug修复时发布。
* 特殊版本（_post1_ 或 _patch1_） 在向后兼容的bug修复时发布，可以是显式或隐式的发布后（post）版本。

## 发布节奏

补丁版本每两周发布一次。发布后（post）版本在补丁版本发布后1-3天发布，并使用与补丁版本相同的分支。
以下是2025年的发布节奏。所有未来的发布日期均为暂定。请注意：发布后（post）版本是可选的。

| 发布日期   | 补丁版本号          | 发布后版本号 |
| ---------- | ------------------- | ------------ |
| 2025年1月  | 0.7.0               | ---          |
| 2025年2月  | 0.7.1, 0.7.2, 0.7.3 | ---          |
| 2025年3月  | 0.7.4, 0.7.5        | ---          |
| 2025年4月  | 0.7.6, 0.7.7        | ---          |
| 2025年5月  | 0.7.8, 0.7.9        | ---          |
| 2025年6月  | 0.7.10, 0.7.11      | ---          |
| 2025年7月  | 0.7.12, 0.7.13      | ---          |
| 2025年8月  | 0.7.14, 0.7.15      | ---          |
| 2025年9月  | 0.7.16, 0.7.17      | ---          |
| 2025年10月 | 0.7.18, 0.7.19      | ---          |
| 2025年11月 | 0.7.20, 0.7.21      | ---          |
| 2025年12月 | 0.7.22, 0.7.23      | ---          |

## 发布分支

每个发布都从一个专用的release分支构建。
* 对于主版本（_major_），次版本（_minor_），补丁版本（_patch_）发布，发布分支的切割在发布前1-2天进行。
* 对于发布后（post）版本，使用之前切出的发布分支。
* 发布构建通过推送到RC标签（如vX.Y.Z-rc1）来触发。这使得我们能够为每个发布构建和测试多个RC。
* 最终标签（Final tag）：vX.Y.Z不会触发构建，但用于发布说明和资源。
* 在分支切割后，我们会监控主分支的任何回滚操作，并将这些操作应用到发布分支。

## 发布的Cherry-Pick标准

在分支切割后，我们会根据明确的标准来决定哪些更改可以被cherry-pick到发布分支中。（注：cherry-pick是指在分支切出后，将某个Pull Request的更改手动合并到发布分支中的过程。）Cherry-pick的范围通常受到严格限制，以确保团队有充足时间对稳定的代码库进行测试。

* 回归修复（Regression fixes）:解决与最近发布版本（例如0.7.0对于0.7.1发布）相比的功能/性能回归问题。
* 关键修复（Critical fixes）：针对严重问题的关键修复，例如静默的错误结果、向后兼容性破坏、程序崩溃、死锁，以及（较大的）内存泄漏等。
* 新功能修复（Fixes to new features）：修复在最近一个正式发布版本（例如0.7.0对0.7.1）中引入的新功能所存在的问题。
* 文档改进（Documentation improvements）
* 发布分支专属变更（Release branch specific changes）：例如更新版本标识符或修复 CI 配置等。

请注意：**Cherry-pick不允许包含任何新功能开发**。所有要被cherry-pick的PR必须首先合并到主干（trunk）分支，唯一的例外是仅适用于发布分支的特定更改。

## 人工验证

### 端到端性能验证（E2E Performance Validation）

在发布之前，我们会执行端到端（end-to-end）的性能验证，以确保未引入任何性能退化。该验证通过PyTorch CI上的[vLLM基准测试工作流（vllm-benchmark workflow）](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml)进行。

**当前覆盖范围：**
* 模型：Llama3, Llama4, Mixtral
* 硬件: NVIDIA H100和AMD MI300x
* _注意: 覆盖范围可能会根据新模型的发布和硬件的可用性进行调整_

**性能验证流程：**
**步骤1：获取访问权限**
申请对[pytorch/pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing)仓库的写入权限以便运行基准测试工作流.

**步骤2：复习基准测试程序配置**
熟悉当前的基准测试设置:

* [CUDA配置（CUDA setup）](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/cuda)
* [ROCm配置（ROCm setup）](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/rocm)

**步骤3：运行基准测试程序**
前往[vLLM基准测试工作流（vllm-benchmark workflow）](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml)并配置以下参数:

* **vLLM 分支（vLLM branch）**: 设置为发布分支(例如 `releases/v0.9.2`)
* **vLLM 提交（vLLM commit）**: 设为候选发布（RC）的提交哈希值

**步骤4：查看测试结果**
工作流完成后，基准测试结果可以在[vLLM 基准测试仪表盘（vLLM benchmark dashboard）](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm)下的对应分支和提交处查看。

**步骤5：性能对比**
比较当前结果与前一个发布版本的，以验证没有性能回归。以下是一个例子，展示了[v0.9.1 与 v0.9.2 的性能对比](https://hud.pytorch.org/benchmark/llms?startTime=Thu%2C%2017%20Apr%202025%2021%3A43%3A50%20GMT&stopTime=Wed%2C%2016%20Jul%202025%2021%3A43%3A50%20GMT&granularity=week&lBranch=releases/v0.9.1&lCommit=b6553be1bc75f046b00046a4ad7576364d03c835&rBranch=releases/v0.9.2&rCommit=a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f&repoName=vllm-project%2Fvllm&benchmarkName=&modelName=All%20Models&backendName=All%20Backends&modeName=All%20Modes&dtypeName=All%20DType&deviceName=All%20Devices&archName=All%20Platforms)。