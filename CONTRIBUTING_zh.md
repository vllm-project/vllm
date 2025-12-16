# 为 vLLM 做贡献

感谢您对为 vLLM 做贡献感兴趣！我们的社区向所有人开放，欢迎各种形式的贡献，无论大小。您可以通过以下几种方式为项目做出贡献：

- 识别并报告任何问题或错误
- 请求或添加对新模型的支持
- 建议或实现新功能
- 改进文档或贡献使用指南

我们也相信社区支持的力量；因此，回答问题、提供 PR 审查和帮助他人也是备受推崇和有益的贡献。

最后，支持我们的最具影响力的方式之一是提高 vLLM 的知名度。在您的博客文章中谈论它，突出它如何推动您的精彩项目。如果您正在使用 vLLM，请在社交媒体上表达您的支持，或者简单地通过为我们的仓库加星标来表示您的赞赏！

## 任务看板

不确定从哪里开始？查看以下链接获取可参与的任务：

- [适合初学者的问题](https://github.com/vllm-project/vllm/labels/good%20first%20issue)
- [精选入门任务](https://github.com/vllm-project/vllm/issues/8133)
- [新模型请求](https://github.com/vllm-project/vllm/labels/new%20model)
- [具有多模态能力的模型](https://github.com/vllm-project/vllm/issues/8134)

## 许可证

请参阅 [LICENSE](LICENSE)。

## 开发

为 vLLM 做贡献的第一步是克隆 GitHub 仓库：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

然后，配置您的 Python 虚拟环境。

仅在 NVIDIA CUDA 上，建议使用 uv（一个非常快速的 Python 环境管理器）来创建和管理 Python 环境。请按照文档安装 uv。安装 uv 后，您可以使用以下命令创建新的 Python 环境：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

如果您只开发 vLLM 的 Python 代码，请使用以下命令安装 vLLM：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

如果您开发 vLLM 的 Python 和 CUDA/C++ 代码，请使用以下命令安装 vLLM：

```bash
uv pip install -e .
```

有关从源代码安装和为其他硬件安装的更多详细信息，请查看您硬件的安装说明，并前往 "从源代码构建 wheel" 部分。

对于迭代 C++/CUDA 内核时的优化工作流，请参阅 [增量编译工作流](https://docs.vllm.ai/en/latest/developer_guides/compile.html#incremental-compilation-workflow) 以获取建议。

> 提示：vLLM 兼容 Python 3.10 到 3.13 版本。然而，vLLM 的默认 Dockerfile 随附 Python 3.12，CI 中的测试（mypy 除外）使用 Python 3.12 运行。因此，我们建议使用 Python 3.12 进行开发，以最大程度减少您的本地环境与我们的 CI 环境冲突的机会。

## 代码 linting

vLLM 使用 pre-commit 来 lint 和格式化代码库。如果您不熟悉 pre-commit，请参阅 [https://pre-commit.com/#usage](https://pre-commit.com/#usage)。设置 pre-commit 非常简单：

```bash
uv pip install pre-commit
pre-commit install
```

现在，每次提交时都会自动运行 vLLM 的 pre-commit 钩子。

> 提示：您可以使用以下命令手动运行 pre-commit 钩子：
> ```bash
> pre-commit run     # 对暂存文件运行
> pre-commit run -a  # 对所有文件运行（--all-files 的简写）
> ```

有些钩子只在 CI 中运行。如果需要，您可以在本地运行它们：

```bash
pre-commit run --hook-stage manual markdownlint
pre-commit run --hook-stage manual mypy-3.10
```

## 文档

MkDocs 是一个快速、简单且非常美观的静态站点生成器，专为构建项目文档而设计。文档源文件以 Markdown 编写，并使用单个 YAML 配置文件 `mkdocs.yaml` 进行配置。

开始使用：

```bash
uv pip install -r requirements/docs.txt
```

> 提示：确保您的 Python 版本与插件兼容（例如，mkdocs-awesome-nav 需要 Python 3.10+）

MkDocs 带有一个内置的开发服务器，可让您在处理文档时预览文档。从仓库的根目录运行：

```bash
mkdocs serve                           # 包含 API 参考（约 10 分钟）
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # 排除 API 参考（约 15 秒）
```

一旦您在日志中看到 "Serving on http://127.0.0.1:8000/"，实时预览就已准备就绪！在浏览器中打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 即可查看。

有关其他功能和高级配置，请参考：

- [MkDocs 文档](https://www.mkdocs.org/)
- [Material for MkDocs 文档](https://squidfunk.github.io/mkdocs-material/)（我们使用的 MkDocs 主题）

## 测试

vLLM 使用 pytest 测试代码库。

```bash
# 安装 CI 中使用的测试依赖项（仅 CUDA）
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# 安装一些常见的测试依赖项（硬件无关）
uv pip install pytest pytest-asyncio

# 运行所有测试
pytest tests/

# 运行单个测试文件的测试并显示详细输出
pytest -s -v tests/test_logger.py
```

### 如果缺少 Python.h，请安装 python3-dev

如果上述任何命令失败并显示 "Python.h: No such file or directory"，请使用 `sudo apt install python3-dev` 安装 python3-dev。

> 警告：目前，仓库并未完全通过 mypy 检查。
> 
> 目前，并非所有单元测试在 CPU 平台上运行时都会通过。如果您无法访问 GPU 平台在本地运行单元测试，请暂时依赖持续集成系统来运行测试。

## 问题

如果您遇到 bug 或有功能请求，请先搜索现有问题，看看是否已经有人报告过。如果没有，请提交一个新问题，并提供尽可能多的相关信息。

> 重要：如果您发现安全漏洞，请按照 [此处](https://docs.vllm.ai/en/latest/community/security.html) 的说明操作。

## 拉取请求和代码审查

感谢您对 vLLM 的贡献！在提交拉取请求之前，请确保 PR 符合以下标准。这有助于 vLLM 保持代码质量并提高审查过程的效率。

### DCO 和 Signed-off-by

为该项目贡献更改时，您必须同意 [DCO](https://developercertificate.org/)。提交必须包含 `Signed-off-by:` 头部，以证明您同意 DCO 的条款。

使用 `git commit -s` 将自动添加此头部。

> 提示：您可以通过 IDE 启用自动签名：
> - PyCharm：在 "Commit and Push..." 窗口中，点击 "Commit" 按钮右侧的 "Show Commit Options" 图标。它会弹出一个窗口，您可以在其中修改 git 作者并启用 "Sign-off commit"。
> - VSCode：打开设置编辑器并启用 "Git: Always Sign Off"（git.alwaysSignOff）字段。

### PR 标题和分类

只有特定类型的 PR 会被审查。PR 标题应适当添加前缀，以指示更改的类型。请使用以下之一：

- [Bugfix] 用于 bug 修复
- [CI/Build] 用于构建或持续集成改进
- [Doc] 用于文档修复和改进
- [Model] 用于添加新模型或改进现有模型。模型名称应出现在标题中
- [Frontend] 用于 vLLM 前端的更改（例如，OpenAI API 服务器、LLM 类等）
- [Kernel] 用于影响 CUDA 内核或其他计算内核的更改
- [Core] 用于核心 vLLM 逻辑的更改（例如，LLMEngine、AsyncLLMEngine、Scheduler 等）
- [Hardware][Vendor] 用于硬件特定的更改。供应商名称应出现在前缀中（例如，[Hardware][AMD]）
- [Misc] 用于不符合上述类别的 PR。请谨慎使用

> 注意：如果 PR 跨越多个类别，请包含所有相关前缀。

### 代码质量

PR 需要满足以下代码质量标准：

- 我们遵守 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html) 和 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)
- 通过所有 linter 检查
- 代码需要有良好的文档，以确保未来的贡献者可以轻松理解代码
- 包含足够的测试，以确保项目保持正确和健壮。这包括单元测试和集成测试

如果 PR 修改了 vLLM 的用户面向行为，请将文档添加到 `docs/`。这有助于 vLLM 用户理解和利用新功能或更改。

### 添加或更改内核

当积极开发或修改内核时，强烈建议使用增量编译工作流以加快构建速度。每个自定义内核需要一个模式和一个或多个实现才能注册到 PyTorch。

- 确保按照 PyTorch 指南注册自定义操作：[自定义 C++ 和 CUDA 操作](https://pytorch.org/tutorials/advanced/cpp_extension.html) 和 [自定义操作手册](https://pytorch.org/docs/stable/notes/extending.html)
- 返回张量的自定义操作需要元函数。元函数应在 Python 中实现和注册，以便自动处理动态维度。有关元函数的描述，请参阅上述文档
- 使用 `torch.library.opcheck()` 测试任何注册操作的函数注册和元函数。请参阅 `tests/kernels` 中的示例
- 更改现有操作的 C++ 签名时，必须更新模式以反映更改
- 如果需要新的自定义类型，请参阅以下文档：[PT2 中的自定义类支持](https://pytorch.org/docs/stable/notes/extending.html#custom-class-support-in-pt2)

### 大更改说明

请保持更改尽可能简洁。对于重大架构更改（>500 LOC，不包括内核/数据/配置/测试），我们希望有一个 GitHub 问题（RFC）来讨论技术设计和理由。否则，我们将为其添加 `rfc-required` 标签，并且可能不会审查该 PR。

### 审查流程预期

vLLM 团队的目标是成为一个透明的审查机器。我们希望使审查过程透明高效，并确保没有贡献者感到困惑或沮丧。然而，vLLM 团队规模很小，因此我们需要优先处理一些 PR。以下是您可以从审查过程中预期的内容：

1. PR 提交后，将分配给审查者。每位审查者将根据他们的专业知识和可用性处理 PR
2. PR 分配后，审查者将每 2-3 天提供状态更新。如果 PR 在 7 天内未被审查，请随时提醒审查者或 vLLM 团队
3. 审查后，如果需要更改，审查者将在 PR 上添加 `action-required` 标签。贡献者应解决评论并提醒审查者重新审查 PR
4. 请在合理的时间内回复所有评论。如果评论不清楚或您不同意建议，请随时要求澄清或讨论建议

请注意，由于计算资源有限，并非所有 CI 检查都会执行。当 PR 准备好合并或需要完整的 CI 运行时，审查者将添加 `ready` 标签。

## 感谢

最后，感谢您抽出时间阅读这些指南，并对为 vLLM 做贡献感兴趣。您的所有贡献都有助于使 vLLM 成为一个为每个人服务的伟大工具和社区！
