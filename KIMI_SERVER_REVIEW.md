# Kimi服务器改动核对草稿

删除服务器不存在的通用sampling trace processor、demo、CPU测试和CI接线。恢复服务器原来的模型输出采集、GPU runner采集入口，以及暂停时调整scheduler budget的分段prefill诊断接口。

只保留主仓现有的grammar/diffusion判断，在其之前放回原采集入口。探针仍是实验专用，会读取设备张量到CPU并写本地文件；不是异步GPU友好的通用采集机制。

这是供作者查看423实际实验改动的草稿，不申请合入。代码中的实验开关、独立helper依赖、同步采集和模型专用路径按服务器保留；不为了满足上游规范另做重构。

共享实现保存在配套verl分支的`examples/kimi_alignment/runtime/`，其中13个Python文件与此次423只读采集逐字节一致。沿用服务器的独立`alignment_runtime`/PYTHONPATH布局；四个仓库需要这份共同依赖。部分入口在判断开关前就导入helper，因此即使关闭对齐开关也必须加入这个PYTHONPATH目录。没有新增包内loader替换或MC2 callback层。模型权重、checkpoint配置及继承环境仍需原实验配置，草稿不是可直接替换423的全套安装包。

原始服务器增量见`KIMI_SERVER_TASK.patch`，逐文件来源与SHA-256见`KIMI_SERVER_REVIEW.json`。主仓已改名的路径和必要的调用衔接在下文列出；除共享helper外，不声称主线文件与服务器整文件相同。服务器未被修改。

历史结果：423的4层、16专家、top16、BF16短测attempt32/33/34每轮两步，每步1024个有效response token，旧分数/rollout分数的差异token数、平均绝对差和最大绝对差均为0；attempt34的old/grad与actor/reference比较也为0。这些结果只属于当时服务器布局。短BI/PD样本不代表完整验收，attempt35在HCCL初始化失败、长轨迹和分段prefill未验收；reward/advantage/grad norm为0，有效更新与同步、93层和性能仍待验证。

本次只做SSH文件读取、SHA-256比对、Python AST语法和源码差异检查。未导入模型/算子包，未运行单测、GPU/NPU测试、模型或训练。旧通用采集机制的20个CPU测试已随该机制撤掉，不能作为现在草稿的测试结果。

FSDPTurbo恢复的96头SFA/LIG兼容分支不受KIMI_ALIGNMENT_RECURRENT_KDA开关控制；96头非shared-K/V的SFA会抛错。这是服务器原行为，不能描述为关闭开关即可保持主仓默认行为。

AI assistance: Codex核对并整理草稿。此前的上游规范建议不作为本轮新增重构任务；人工审阅和当前主线兼容性/数值复验仍未完成。
