# Hook Reload 暂不支持场景清单

hook reload 路径**不做 layerwise 回退**：遇到下列场景时
`initialize_reload` 抛出 `HookReloadUnsupportedError`，由上层决定
重试 / cold load / 终止服务。每项都计划在后续迭代中补齐 hook 支持。

| # | 场景 | 现状 | 计划 |
| --- | --- | --- | --- |
| 1 | LoRA 组合（lora_enabled） | 抛异常 | LoRA 参数对象会被包装，指针语义与 hook 路径冲突，待设计 |
| 2 | 在线量化（checkpoint 非 FP8 序列化，如 bf16→fp8 在线量化） | 抛异常 | 需要 CONVERT buffer（requant）hook，文档 §6.4 |
| 3 | FP8 per-tensor（weight_block_size 为空） | 抛异常 | 原 layerwise selective 路径已随回退取消一并停用；per-tensor hook（含融合 QKV 取 max requant）待实现，见主文档 §8.2 |
| 4 | FP8 block-wise：UE8M0 requant（SM100 DeepGemm，`VLLM_USE_DEEP_GEMM_E8M0=1`） | 已支持 | weight 与 scale 共同 staging，完成后联合 requantize 并写回稳定 runtime storage |
| 5 | FP8 block-wise：weight_scale_refine（TP 切分与 block 不对齐时 scale 在 load 时被细化上采样） | 抛异常 | runtime scale 形状与 checkpoint 不同，需要重采样映射 hook |
| 6 | FP8 block-wise：Humming / B12x / XPU / CPU / ROCm(AITER) 后端 | 抛异常 | 非主线后端，layout 各异，按需逐个补齐 |
| 6a | FP8 block-wise MoE Marlin / dense Marlin 带 bias | 抛异常 | MoE marlin repack 与 bias permute 未映射；dense Marlin block（无 bias）已支持（staging + finish repack） |
| 7 | FP8 MoE：FLASHINFER_TRTLLM（BlockMajorK shuffle） | 抛异常 | 权重本体非视图变换，属于文档 §8.1 #9 后备路径 |
| 8 | 非 FP8 量化方案（AWQ / GPTQ / INT8 等） | 抛异常 | 未审计，按需补齐 |
| 9 | dummy 初始化模型（无 cold-load 观察计划） | 抛异常 | hook 计划来自 cold load 观察；dummy 启动后需先做一次真实 cold load |
| 10 | EPLB 开启的 MoE | 未审计 | expert_map 使 expert_id 映射动态化，待验证 |
