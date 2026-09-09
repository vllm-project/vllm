# Expert重みのGPUキャッシュ：速度測定の再現手順

Qwen3.8 FlashNext NVFP4に、準備用の要求と速度測定用の要求を順番に送ります。
各要求ではソフトウェアの不具合報告を読み、修正案と検証計画を生成します。

## ファイルと要求の内容

| ファイル | 内容 |
| --- | --- |
| [benchmark.py](benchmark.py) | HTTP要求、ストリーム保存、生成速度の集計 |
| [pair.json](pair.json) | 実測に使用したプロンプト全文、固定トークン列、出典 |
| [prompts.md](prompts.md) | プロンプト全文の読みやすい表示 |
| [provenance.json](provenance.json) | 実測コードの版、元ファイルのSHA256、移植差分 |
| [LICENSE.prompts](LICENSE.prompts) | プロンプトの出典に付属するMITライセンス |

| 送信順 | 用途 | 不具合報告 | 入力トークン数 |
| --- | --- | --- | --- |
| 1回目 | 準備用（結果のroleはwarmup） | ファイル選択ボタンの「Choose File」を「Choose file」に修正する課題（28096_836） | 1070 |
| 2回目 | 速度測定用（roleはmeasure） | 言語設定を変更したとき「Link sent!」表示も更新する課題（18827_741） | 753 |

出典は[OpenAI frontier-evals](https://github.com/openai/frontier-evals/tree/51052cede8cc608f95bb00346635e03759013e5a)のSWE-Lancerです。
既存の動作確認用課題から選んだ2件で、測定対象は修正案の文章生成です。
実際のコード編集・公式採点を行う品質試験は別の手順です。

## 実測に使用したコードと環境

| 用途 | コードの版 |
| --- | --- |
| 比較の土台となるvLLM main | `a97dacb7106ee49f39f3d1fc6ae1800ff724e01d` |
| Expert重みのGPUキャッシュの実装（[fork PR #48](https://github.com/01554/vllm/pull/48)） | `5fbc240ba5ddec82a10362340ac77339a1c24017` |
| PLEの読み出しと生成計算を重ねる実装（[fork PR #46](https://github.com/01554/vllm/pull/46)、[上流PR #54129](https://github.com/vllm-project/vllm/pull/54129)を前提とする差分） | `4f859de9d0f55760b50358aee4834e6966e13bc8` |
| 上記機能を組み合わせ、以下の速度を測定した版 | `7dedc6d8d9b178b60f6a5b32f03d677145982441` |

サーバーは実測版のPythonソース、上記mainからビルドしたwheel、別途ビルドした
`_ple_memops`拡張を組み合わせて動かしました。このディレクトリは測定後に追加した
クライアント用ファイルです。`pair.json`は実測ファイルのbyteコピーです。

RTX 6000 Adaの48GBに収める構成の検証を目的として、手元の
RTX PRO 6000 Blackwell Max-Q（96GiB）上で別プロセスにGPUメモリを確保させ、
サーバーに利用可能な容量を48GiBにして測定しました。以下はこのGPU上の実測値です。
ホスト側のコンテナのメモリ上限は100GiBでした。

GPUキャッシュは48層それぞれ258 expert行、約32GiBです。
容量制限は外部プロセスで設定します。下記の`gpu-memory-utilization`は物理GPU容量に
対するvLLMの予算比率です。クライアント実行前に容量とサーバー起動状態を確認してください。

## サーバーの起動設定

上の実測版と同じ機能をビルドした環境で、チェックポイントのパスを指定します。
`VLLM_USE_BREAKABLE_CUDAGRAPH`は未設定（自動選択）で測定しました。

```bash
export CHECKPOINT=/data/models/Qwen3.8-Flash-Next-NVFP4-nvidia
unset VLLM_USE_BREAKABLE_CUDAGRAPH
export VLLM_DEBUG_WORKSPACE=1 VLLM_LOGGING_LEVEL=DEBUG
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1,pinned_max_cached_size_mb:1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLE_MMAP=1 VLLM_PLE_MMAP_DEFERRED=1
export VLLM_PLE_MMAP_PREWARM=0 VLLM_PLE_MMAP_PINNED=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CHECKPOINT" --served-model-name flashnext \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --quantization modelopt --dtype bfloat16 --moe-backend marlin \
  --moe-expert-pool-rows 258 --language-model-only \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --no-enable-flashinfer-autotune --gpu-memory-utilization 0.4548806288994517 \
  --safetensors-load-strategy lazy \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --reasoning-parser qwen3 --generation-config vllm
```

速度測定のコンテキスト上限は4096です。別途実施した品質試験では32768を使いました。

## クライアントの実行

サーバーの準備完了後、次を1回実行します。クライアントはPython標準ライブラリで動きます。

```bash
.venv/bin/python benchmarks/expert_pool/benchmark.py \
  --base-url http://127.0.0.1:8000 --model flashnext \
  --label fresh-0 --output results/fresh-0.jsonl
```

送信先は`/v1/completions`です。`pair.json`の固定トークン列を送信するので、
チェックポイントのtokenizerが`pair.json`の`tokenization`に記録したSHA256と
一致することを確認してください。要求には`temperature=0`、`top_p=1`、`seed=0`、
`max_tokens=2048`を指定し、thinkingを無効にしたチャットテンプレートのトークン列を使います。
`tokenization.pair_sha256`はトークン列追加前の資料のハッシュで、ファイル全体のハッシュは
`provenance.json`にあります。

3回の測定では、**毎回サーバーを終了して新しいプロセスで起動し、準備用→測定用を1組送信**します。
出力名を`fresh-0.jsonl`、`fresh-1.jsonl`、`fresh-2.jsonl`と変えます。
実測はこの順で3組を実行し、2回目の要求の速度3値から中央値を求めました。
クライアントは既存の出力ファイルを保護し、HTTPエラーや不完全なストリームを保存して停止します。
失敗した測定も結果として保持してください。

## 指標と実測値

生成速度は、usageの生成トークン数とクライアント側の受信時刻から計算します。

```text
decode_tok_s = (completion_tokens - 1) / (最後の本文受信時刻 - 最初の本文受信時刻)
```

`first_token_s`は要求開始から最初の本文受信までの時間です。
`e2e_tok_s`は要求全体の所要時間あたりの生成トークン数です。
ストリームの1イベントに複数トークンが入ることがあるため、いずれもクライアント観測の値です。
生のSSE、送信body、usage、finish reason、本文とSHA256も同じJSONLに保存します。

| 新しいサーバーでの実行 | 2回目の要求の生成速度 |
| --- | --- |
| 1回目 | 63.1882 tok/s |
| 2回目 | 62.7087 tok/s |
| 3回目 | 63.6519 tok/s |
| 中央値 | **63.1882 tok/s** |

これは表の実測版で各機能を組み合わせた値です。各要求の終了理由は`stop`でした。
容量は起動中・生成中の標本で記録し、プロセス終了コード0とOOMKilled=falseを確認しました。

## クライアントの動作確認

```bash
.venv/bin/python -m unittest discover -s benchmarks/expert_pool -p 'test_*.py'
```
