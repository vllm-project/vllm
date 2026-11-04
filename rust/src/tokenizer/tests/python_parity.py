# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regenerate the Python frontend's token ids for `python_parity.rs`.

Run from a vLLM checkout with vLLM installed (CPU is enough):

    python rust/src/tokenizer/tests/python_parity.py [KEY ...]

For each tokenizer below, this loads the tokenizer the way the Python frontend
does (`vllm.tokenizers.get_tokenizer` with the recipe's tokenizer mode) and
encodes every case of `python_parity/corpus.json` with the arguments
`TokenizeParams.get_encode_kwargs` passes for a prompt without a length limit.
The ids go to `python_parity/<KEY>.json` together with the pinned model
revision and the files the Rust test loads. Only tokenizer and config files
are downloaded, never weights.
"""

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import huggingface_hub
import tokenizers
import transformers

from vllm.renderers.params import TokenizeParams
from vllm.tokenizers import get_tokenizer

SNAPSHOT_DIR = Path(__file__).parent / "python_parity"


@dataclass(frozen=True)
class TokenizerCase:
    """One tokenizer under test, pinned to a Hub revision."""

    key: str
    model: str
    revision: str
    # Files the Rust test downloads; the first one is the tokenizer itself.
    files: tuple[str, ...]
    tokenizer_mode: str = "auto"


# One tokenizer per distinct tokenizer pipeline among the recipes that default
# to the Rust frontend, plus a published `tokenizer.json` with a truncation block.
TOKENIZERS = [
    TokenizerCase(
        "qwen35",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        ("tokenizer.json", "tokenizer_config.json"),
    ),
    TokenizerCase(
        "glm52",
        "zai-org/GLM-5.2-FP8",
        "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
        ("tokenizer.json", "tokenizer_config.json"),
    ),
    TokenizerCase(
        "deepseek_v4",
        "deepseek-ai/DeepSeek-V4-Flash",
        "60d8d70770c6776ff598c94bb586a859a38244f1",
        ("tokenizer.json", "tokenizer_config.json"),
        tokenizer_mode="deepseek_v4",
    ),
    TokenizerCase(
        "hy_v3",
        "tencent/Hy3",
        "a960ebc3da325ba167f069f76c41eb62c9280d22",
        ("tokenizer.json", "tokenizer_config.json"),
    ),
    TokenizerCase(
        "minimax_m25",
        "MiniMaxAI/MiniMax-M2.5",
        "f710177d938eff80b684d42c5aa84b382612f21f",
        ("tokenizer.json", "tokenizer_config.json"),
    ),
    TokenizerCase(
        "kimi_k3",
        "moonshotai/Kimi-K3",
        "f831ab66814297da540d832a5235f8e904f29d06",
        ("tiktoken.model", "tokenizer_config.json", "config.json"),
    ),
    TokenizerCase(
        "muse_spark",
        "amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4",
        "c7bc74fd64371315a091bc15f9723882eead0dc1",
        ("tokenizer.json", "tokenizer_config.json"),
    ),
    TokenizerCase(
        "deepseek_r1_truncation",
        "amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4",
        "a1b11449f6467175ad6851e13590ac3424edfb22",
        ("tokenizer.json", "tokenizer_config.json"),
    ),
]

# Besides the Rust test's files, the Python loader reads the model config and
# remote tokenizer code.
PYTHON_ONLY_FILES = ["config.json", "special_tokens_map.json", "*.py"]


def case_text(corpus: dict, case: dict, special_token: str) -> str:
    """Build one case's text, as `python_parity.rs` does."""
    prefix = corpus["prefixes"][case["prefix"]]
    suffix = case["suffix"].replace(corpus["special_token_placeholder"], special_token)
    return prefix["text"] * prefix["repeat"] + suffix


def download(case: TokenizerCase) -> Path:
    return Path(
        huggingface_hub.snapshot_download(
            case.model,
            revision=case.revision,
            allow_patterns=[*case.files, *PYTHON_ONLY_FILES],
        )
    )


def encode_corpus(tokenizer, corpus: dict, special_token: str) -> dict[str, list]:
    """Encode every case the way the Python frontend encodes a text prompt."""
    ids = {}
    for case in corpus["cases"]:
        params = TokenizeParams(
            max_total_tokens=None, add_special_tokens=case["add_special_tokens"]
        )
        text = case_text(corpus, case, special_token)
        ids[case["name"]] = tokenizer(text, **params.get_encode_kwargs())["input_ids"]
    return ids


def load(case: TokenizerCase, path: Path):
    return get_tokenizer(
        path, tokenizer_mode=case.tokenizer_mode, trust_remote_code=True
    )


def check_overrides_ignored(
    case: TokenizerCase, path: Path, corpus: dict, special_token: str, ids: dict
) -> None:
    """Check that the Python frontend ignores `tokenizer.json`'s truncation and
    padding, which the Rust test relies on to reuse the same ids."""
    if case.files[0] != "tokenizer.json":
        return
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "model"
        shutil.copytree(path, copy)
        target = copy / "tokenizer.json"
        tokenizer_json = json.loads(target.read_text())
        tokenizer_json.update(corpus["tokenizer_json_overrides"])
        # Hub cache files are read-only; replace the copy instead of writing to it.
        target.unlink()
        target.write_text(json.dumps(tokenizer_json))
        overridden = encode_corpus(load(case, copy), corpus, special_token)
    if overridden != ids:
        raise AssertionError(f"{case.key}: truncation or padding changed the ids")


def snapshot(case: TokenizerCase, special_token: str, ids: dict) -> str:
    """Render the snapshot with one line per case, so diffs stay per case."""
    header = {
        "model": case.model,
        "revision": case.revision,
        "files": list(case.files),
        "tokenizer_mode": case.tokenizer_mode,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "special_token": special_token,
    }
    lines = [
        f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},"
        for k, v in header.items()
    ]
    id_lines = [f"    {json.dumps(name)}: {json.dumps(v)}" for name, v in ids.items()]
    return (
        "{\n"
        + "\n".join(lines)
        + '\n  "ids": {\n'
        + ",\n".join(id_lines)
        + "\n  }\n}\n"
    )


def generate(case: TokenizerCase, corpus: dict) -> None:
    path = download(case)
    tokenizer = load(case, path)
    special_token = tokenizer.eos_token
    ids = encode_corpus(tokenizer, corpus, special_token)
    check_overrides_ignored(case, path, corpus, special_token, ids)
    out = SNAPSHOT_DIR / f"{case.key}.json"
    out.write_text(snapshot(case, special_token, ids))
    print(f"wrote {out} ({sum(map(len, ids.values()))} ids)")


def main(keys: list[str]) -> int:
    corpus = json.loads((SNAPSHOT_DIR / "corpus.json").read_text())
    selected = [case for case in TOKENIZERS if not keys or case.key in keys]
    unknown = set(keys) - {case.key for case in TOKENIZERS}
    if unknown:
        print(f"unknown tokenizer keys: {sorted(unknown)}", file=sys.stderr)
        return 2
    for case in selected:
        generate(case, corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
