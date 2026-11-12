# 开发与贡献

## 开始前

先阅读仓库根目录的 [`AGENTS.md`](../../AGENTS.md)、相关目录中的 `AGENTS.md`，以及[贡献指南](../contributing/README.md)。它们定义环境、代码风格、测试选择、文档要求和领域特定限制。尤其是 C++/CUDA、Rust、模型、CI 等子系统，不应只凭通用 Python 经验修改。

项目开发环境使用 `uv` 管理 Python 依赖；仓库约定禁止使用系统 `python3` 或裸 `pip`。安装方式依目标硬件和改动范围不同，按[安装指南](../getting_started/installation/README.md)和[贡献指南](../contributing/README.md)选择。

## 建议的改动闭环

1. **找行为入口**：确定用户如何触发行为，是 CLI/API、Python `LLM`、配置、模型输入还是设备算子。
2. **找决策点**：沿转发路径走到真正计算、选择、分配或改变状态的代码。
3. **找契约和测试**：查看设计文档、接口协议和相邻测试，明确输入输出及失败场景。
4. **做最小修改**：遵循模块已有抽象；避免为了一个局部需求重写相邻子系统。
5. **先跑定向验证**：从单测、类型/格式检查或小规模集成测试开始，再根据改动风险扩展。
6. **同步面向用户的文档**：CLI 参数、配置字段、硬件/模型支持和默认值变化都需要更新对应文档。

## 按贡献类型选入口

### 新模型或模型适配

从[模型开发指南](../contributing/model/README.md)开始，依次阅读[基础模型实现](../contributing/model/basic.md)、[注册模型](../contributing/model/registration.md)及对应任务文档。多模态模型另看[多模态模型开发](../contributing/model/multimodal.md)。最终验证方式参考[模型测试指南](../contributing/model/tests.md)。

### Python 核心功能

先找功能所属子系统和 `tests/` 下对应测试；尽量通过真实对象和最轻真实路径验证。用户可见行为应覆盖实际公开入口，而不只断言内部方法被调用。项目测试规则见[`tests/AGENTS.md`](../../tests/AGENTS.md)。

### CUDA、Triton 或其他 kernel

正确性测试与性能评估是不同目标：先建立正确性覆盖，再在代表性硬件/shape 上做微基准。C++/CUDA 编译按[增量编译工作流](../contributing/incremental_build.md)执行；Triton kernel 还应遵循仓库相应的 kernel 开发规范和测试约定。

### 文档

文档源文件使用 Markdown，站点由 MkDocs 构建；导航规则在 `docs/.nav.yml`。修改用户可见说明时，确认参数名、默认值和行为与当前实现一致，并检查相对链接和代码示例。

## 测试选择原则

测试层级应匹配行为和风险：

- 算法/纯 Python 逻辑：模块级单测。
- 配置解析和默认值：配置测试。
- 模型输入、权重加载或真实输出：模型集成测试。
- API/CLI 可见行为：通过对应公开入口验证。
- 并行或通信协议：对应分布式测试。
- Kernel 行为：设备正确性测试；性能变化另做 benchmark。

CI 选择规则、marker、真实权重要求和测试质量标准以[`tests/AGENTS.md`](../../tests/AGENTS.md)为准。

## 提交前检查

- 查看完整 diff，确认没有无关格式化或意外文件。
- 运行受影响文件的 lint/format 与最相关测试，并在 PR 说明中记录命令和结果。
- 用户可见变更同步更新文档；模型输出或性能相关变更按项目要求补充评估数据。
- PR 前按仓库规则运行 [`pr-checklist`](../.agents/skills/pr-checklist/SKILL.md) 所指的检查；该 skill 文件实际位于仓库根目录 `.agents/skills/`，从本文档的正确相对路径为 `../../.agents/skills/pr-checklist/SKILL.md`。

更多开发背景见[贡献指南](../contributing/README.md)及[模型开发文档](../contributing/model/README.md)。