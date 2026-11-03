# 独立 vLLM Weight Update Client

这个目录提供不依赖 RL 训练框架的权重发送客户端：读取已经准备好的本地
Hugging Face safetensors checkpoint，生成 canonical tensor manifest，并通过
vLLM 原生 NCCL weight-transfer engine 发送给服务端。

当前 active path 只有 `checkpoint_passthrough`。publisher 不做量化、请求、sleep、
wake、KV/prefix-cache reset、LoRA 或 RL 生命周期控制；可选 shell wrapper 只编排
这些 lifecycle API。MTP 只有显式传入 `--enable-mtp` 才会执行第二个独立 draft
update；独立 DraftModel 使用单独的 draft manifest 和 `--enable-draft-update`。
禁止隐式从 Hugging Face 下载，必须显式传入本地 checkpoint。

## 文件

| 文件 | 用途 |
|---|---|
| `hf_checkpoint_nccl_publisher.py` | checkpoint manifest、canonical source、stateful NCCL publisher |
| `run_vllm_weight_update.py` | 只执行一次 checkpoint NCCL weight update 的最小 CLI |
| `run_vllm_lifecycle.sh` | 一次调用完成 VIME lifecycle 和 publisher 的 shell wrapper |
| `validation/` | GSM8K、fixed-token oracle 以及更新前后结果比较工具 |
| `provenance/write_run_sidecars.py` | 为结果目录生成 provenance `.meta.json` |

## 服务端合同

服务端必须使用 NCCL backend，并启用 `packed`；packed buffer 的大小和 buffer
数量使用 vLLM 默认值，更新期间 inference worker world size 不变。示例：

```bash
--weight-transfer-config \
'{"backend":"nccl","packed":true}'
```

client 容器和服务端容器必须使用同一 vLLM runtime。Docker 使用 `--gpus all`，
容器内再用 `CUDA_VISIBLE_DEVICES` 限制实际计算 GPU。
publisher trainer rank 与 serving rank 必须使用不同的物理 GPU。

## 最小运行命令

checkpoint 必须预先放在 publisher 所在节点的本机 NVMe 上，并通过
`--checkpoint-path` 指定；不要直接从 Lustre 读取。direct-file H2D 依赖本地
文件的 mmap 和连续顺序读取，使用 Lustre 会重新引入远端 I/O 和 page fault
开销，抵消这项优化的收益。

```bash
python run_vllm_weight_update.py \
  --base-url http://<server-host>:8000 \
  --model <local-model-path> \
  --revision <revision> \
  --checkpoint-path <local-safetensors-checkpoint> \
  --device cuda:0 \
  --output <result-dir>/weight-update.json
```

性能实验可显式设置：

```bash
  --expert-tensor-order lexical \
  --direct-file-expert-h2d
```

`lexical` 仍保持 non-expert 在前、每个 expert layer 完整成组，只调整层内
tensor 顺序。对于按名称词典序写出的 safetensors，它能使读取顺序匹配文件中的
物理 offset，避免 `0, 1, 2, ...` 自然排序造成的跨文件区间跳读。
`--direct-file-expert-h2d` 在 safetensors payload 物理连续时，通过 mmap 将完整
expert layer 一次复制到 GPU，再返回各 tensor 的 storage-sharing view；不满足
连续布局时自动退回逐 tensor 路径。

如果 RL rollout 明确启用了 native MTP，才增加：

```bash
  --enable-mtp
```

独立 DraftModel 使用另一份本地 checkpoint，并在 main transaction 成功后
执行独立的 draft transaction：

```bash
  --enable-draft-update \
  --draft-model <draft-model-name> \
  --draft-revision <draft-revision> \
  --draft-checkpoint-path <local-draft-checkpoint>
```

两种模式的顺序分别是：

```text
MTP:    main manifest → start_weight_update → buckets → finish
        → start_draft_weight_update → buckets → finish
DraftModel: main manifest → start_weight_update → buckets → finish
        → draft manifest → start_draft_weight_update → buckets → finish
```

DraftModel checkpoint 里的共享 embedding 和未使用的输出头由 draft loader 自己
处理；publisher 不再复制 loader 的过滤规则，只传输该 checkpoint 的完整
manifest，并分别记录 main/draft 的 tensor 数量、字节数和 manifest hash。

调用前服务必须已经处于可更新状态。该命令只执行
`start_weight_update → metadata/update buckets → finish_weight_update`，不会
替调用方 pause、sleep、清 cache、发送请求或 resume。

## 可选 lifecycle wrapper

一次调用完成普通更新，严格执行
`pause(keep) → reset prefix cache → publisher → resume`：

```bash
./run_vllm_lifecycle.sh http://<server-host>:8000 -- \
  .venv/bin/python run_vllm_weight_update.py <publisher arguments>
```

需要完整 colocated/offload cycle 时仍只调用一次：可选 abort 在最前面；随后
`reset → sleep(2) → wake(weights) → pause → reset → publisher → resume →
wake(kv_cache)`。VIME 只有在丢弃未完成 rollout 时才执行 abort，因此必须显式启用。

```bash
./run_vllm_lifecycle.sh http://<server-host>:8000 \
  --abort-inflight --offload -- \
  .venv/bin/python run_vllm_weight_update.py <publisher arguments>
```

publisher 失败时不执行 resume 或 KV wake，服务保持 fail-closed；脚本没有任何
定时 sleep、重试或 inference 请求。

## 代码与 VIME 的边界

publisher 保持 VIME 的 NCCL 事务顺序：先发送 non-expert buckets，再发送完整的
expert-layer buckets；每个 bucket 先 metadata RPC，再调用 vLLM 的
`packed_nccl_broadcast_producer`，所有参数、dtype、shape、字节数和顺序校验
通过后才 finish。
source 是唯一的 canonical HF producer，NCCL 是 sink。任何失败都不调用 finish，
避免把部分更新报告为成功。

运行时版本、镜像 digest 和 worktree commit 必须由每次实验的 provenance 记录，
不再由本目录固定 patch 或历史材料提供。
