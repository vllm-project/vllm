# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import os
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM

DEFAULT_MODEL = "nghuyong/ernie-3.0-xbase-zh"

SUPPORTED_MODES = {
    "all",
    "embed",
    "token_embed",
    "classify",
    "score",
    "token_classify",
}

EMBED_PROMPTS = [
    "今天天气不错，适合出去散步。",
    "vLLM 支持高吞吐的推理服务。",
]
TOKEN_EMBED_PROMPTS = ["百度的文心 ERNIE 是中文模型。"]
SEQ_CLS_PROMPTS = [
    "这家餐厅味道很好，服务也很周到。",
    "产品一般，没有特别亮点。",
]
SCORE_QUERY = "如何提高代码可维护性？"
SCORE_DOCS = [
    "通过统一代码风格、补充测试和模块化设计可以提高可维护性。",
    "今天天气晴朗，适合进行户外运动。",
]
TOKEN_CLS_PROMPTS = ["李彦宏在北京会见了百度研发团队。"]

CUSTOM_KEYS = {
    "model",
    "mode",
    "embed_model",
    "seq_cls_model",
    "score_model",
    "token_cls_model",
    "use_tqdm",
}

def _maybe_force_platform_from_env() -> None:
    forced_device = os.getenv("VLLM_TARGET_DEVICE")
    if not forced_device:
        return

    device_map = {
        "cuda": "vllm.platforms.cuda.CudaPlatform",
        "cpu": "vllm.platforms.cpu.CpuPlatform",
        "xpu": "vllm.platforms.xpu.XPUPlatform",
        "tpu": "vllm.platforms.tpu.TpuPlatform",
    }
    device_key = forced_device.lower()
    qualname = device_map.get(device_key)
    if not qualname:
        return

    import vllm.platforms as platforms

    platforms.builtin_platform_plugins[device_key] = lambda: qualname


def parse_args() -> Namespace:
    _maybe_force_platform_from_env()

    from vllm import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    try:
        parser = EngineArgs.add_cli_args(parser)
    except RuntimeError as exc:
        if "Failed to infer device type" not in str(exc):
            raise
        if not os.getenv("VLLM_TARGET_DEVICE"):
            raise
        _maybe_force_platform_from_env()
        parser = EngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default="all",
        help="Which Ernie example to run.",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_MODEL,
        help="Model for ErnieModel embedding examples.",
    )
    parser.add_argument(
        "--seq-cls-model",
        default=DEFAULT_MODEL,
        help="Model for ErnieForSequenceClassification classify example.",
    )
    parser.add_argument(
        "--score-model",
        default=DEFAULT_MODEL,
        help="Model for ErnieForSequenceClassification score example.",
    )
    parser.add_argument(
        "--token-cls-model",
        default=DEFAULT_MODEL,
        help="Model for ErnieForTokenClassification example.",
    )
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Enable tqdm progress bars for vLLM API calls.",
    )

    # Keep examples stable on low-resource machines.
    parser.set_defaults(
        runner="pooling",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=8,
    )
    return parser.parse_args()


def base_llm_kwargs(args: Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if k not in CUSTOM_KEYS}


@contextmanager
def managed_llm(
    *,
    model: str,
    args: Namespace,
    hf_overrides: dict | None = None,
) -> Iterator[LLM]:
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory

    llm = None
    try:
        llm_kwargs = base_llm_kwargs(args)
        if hf_overrides is not None:
            llm_kwargs["hf_overrides"] = hf_overrides
        llm = LLM(model=model, **llm_kwargs)
        yield llm
    finally:
        del llm
        cleanup_dist_env_and_memory()


def print_title(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_note(note: str) -> None:
    print(f"说明: {note}")


@contextmanager
def _timer(title: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        print(f"{title} completed in {elapsed:.3f}s")


def _run_embed_example(llm: LLM, *, use_tqdm: bool) -> None:
    prompts = EMBED_PROMPTS
    with _timer("embed()"):
        outputs = llm.embed(prompts, use_tqdm=use_tqdm)

    print_title("Ernie embed() 示例")
    for prompt, output in zip(prompts, outputs):
        emb = output.outputs.embedding
        print(f"Prompt: {prompt!r}")
        print(f"Embedding dim: {len(emb)}")
        print(f"Embedding[:8]: {emb[:8]}")
        print("-" * 88)


def _run_token_embed_example(llm: LLM, *, use_tqdm: bool) -> None:
    prompts = TOKEN_EMBED_PROMPTS
    with _timer("encode(token_embed)"):
        outputs = llm.encode(prompts,
                             pooling_task="token_embed",
                             use_tqdm=use_tqdm)

    print_title("Ernie token_embed 示例")
    for prompt, output in zip(prompts, outputs):
        token_vectors = output.outputs.data
        print(f"Prompt: {prompt!r}")
        print(f"Token count: {len(output.prompt_token_ids)}")
        print(f"Token embedding shape: {tuple(token_vectors.shape)}")
        print("-" * 88)


def run_classify_example(args: Namespace) -> None:
    with managed_llm(
        model=args.seq_cls_model,
        args=args,
        hf_overrides={
            "architectures": ["ErnieForSequenceClassification"],
            "num_labels": 2,
            "id2label": {0: "negative", 1: "positive"},
            "label2id": {"negative": 0, "positive": 1},
        },
    ) as llm:
        prompts = SEQ_CLS_PROMPTS
        with _timer("classify()"):
            outputs = llm.classify(prompts, use_tqdm=args.use_tqdm)

        print_title("Ernie classify() 示例")
        print_note("这里使用基础模型 + 覆盖架构，分类头是随机初始化，仅用于验证 API 链路。")
        for prompt, output in zip(prompts, outputs):
            probs = output.outputs.probs
            print(f"Prompt: {prompt!r}")
            print(f"Class probabilities: {probs}")
            print("-" * 88)


def run_score_example(args: Namespace) -> None:
    with managed_llm(
        model=args.score_model,
        args=args,
        hf_overrides={
            "architectures": ["ErnieForSequenceClassification"],
            "num_labels": 1,
            "id2label": {0: "relevance"},
            "label2id": {"relevance": 0},
        },
    ) as llm:
        query = SCORE_QUERY
        documents = SCORE_DOCS
        with _timer("score()"):
            outputs = llm.score(query, documents, use_tqdm=args.use_tqdm)

        print_title("Ernie score() 示例")
        print_note("这里使用基础模型 + 覆盖架构，分类头是随机初始化，仅用于验证 score API。")
        for document, output in zip(documents, outputs):
            print(f"Pair: {[query, document]!r}")
            print(f"Score: {output.outputs.score}")
            print("-" * 88)


def _safe_labels(id2label: dict | None, preds: Iterable[int]) -> list[str]:
    if not id2label:
        return [str(x) for x in preds]
    return [str(id2label.get(int(x), int(x))) for x in preds]


def run_token_classify_example(args: Namespace) -> None:
    with managed_llm(
        model=args.token_cls_model,
        args=args,
        hf_overrides={
            "architectures": ["ErnieForTokenClassification"],
            "num_labels": 5,
            "id2label": {
                0: "O",
                1: "B-PER",
                2: "I-PER",
                3: "B-ORG",
                4: "I-ORG",
            },
            "label2id": {
                "O": 0,
                "B-PER": 1,
                "I-PER": 2,
                "B-ORG": 3,
                "I-ORG": 4,
            },
        },
    ) as llm:
        prompts = TOKEN_CLS_PROMPTS
        with _timer("encode(token_classify)"):
            outputs = llm.encode(prompts,
                                 pooling_task="token_classify",
                                 use_tqdm=args.use_tqdm)
        tokenizer = llm.get_tokenizer()
        id2label = llm.llm_engine.vllm_config.model_config.hf_config.id2label

        print_title("Ernie token_classify 示例")
        print_note("这里使用基础模型 + 覆盖架构，分类头是随机初始化，仅用于验证 token_classify API。")
        for prompt, output in zip(prompts, outputs):
            logits = output.outputs.data
            pred_ids = logits.argmax(dim=-1).tolist()
            labels = _safe_labels(id2label, pred_ids)
            tokens = tokenizer.convert_ids_to_tokens(output.prompt_token_ids)

            print(f"Prompt: {prompt!r}")
            print(f"Logits shape: {tuple(logits.shape)}")
            for token, label in zip(tokens, labels):
                if token not in tokenizer.all_special_tokens:
                    print(f"{token:16} -> {label}")
            print("-" * 88)


def run_embed_examples(args: Namespace, *, mode: str) -> None:
    with managed_llm(
        model=args.embed_model,
        args=args,
        hf_overrides={"architectures": ["ErnieModel"]},
    ) as llm:
        if mode in ("all", "embed"):
            _run_embed_example(llm, use_tqdm=args.use_tqdm)
        if mode in ("all", "token_embed"):
            _run_token_embed_example(llm, use_tqdm=args.use_tqdm)


def main(args: Namespace) -> None:
    if args.mode == "all":
        run_embed_examples(args, mode="all")
        run_classify_example(args)
        run_score_example(args)
        run_token_classify_example(args)
    else:
        if args.mode in ("embed", "token_embed"):
            run_embed_examples(args, mode=args.mode)
        elif args.mode == "classify":
            run_classify_example(args)
        elif args.mode == "score":
            run_score_example(args)
        elif args.mode == "token_classify":
            run_token_classify_example(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main(parse_args())
