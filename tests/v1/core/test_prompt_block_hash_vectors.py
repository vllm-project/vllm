# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin the engine's prompt block hashes that the Rust frontend reproduces.

The Rust frontend computes prompt block hashes itself (see
`rust/src/engine-core-client/src/prompt_block_hasher.rs`), and they must be
byte-identical to the engine's. Its unit tests check them against the vectors
in `VECTORS`; this test checks that the engine still produces those same
vectors, so a change to the engine's hashing fails here instead of silently
splitting the two apart.

Cases marked `frontend_fallback` require Rust to omit every hash; their
expected hashes still pin the engine's complete chain.

If the change is intended, update the Rust hasher to match, then regenerate
the vectors by running this file directly:

    .venv/bin/python tests/v1/core/test_prompt_block_hash_vectors.py
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import regex as re

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequest, PrefixHashConfig
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

REPO_ROOT = Path(__file__).resolve().parents[3]
VECTORS = (
    REPO_ROOT
    / "rust"
    / "src"
    / "engine-core-client"
    / "src"
    / "tests"
    / "prompt_block_hash_vectors.json"
)

HASH_FNS: dict[str, Callable[[Any], bytes]] = {
    "sha256": sha256,
    "sha256_cbor": sha256_cbor,
}

# Token ids at every encoding-width boundary of both the pickle and CBOR
# encodings.
EDGE_TOKENS = [0, 1, 23, 24, 255, 256, 65535, 65536, 151_645, 0x7FFF_FFFF]


def _case_inputs() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    plain_prompts = [
        (16, EDGE_TOKENS * 3 + [7, 8]),
        (1, [0, 300, 70000]),
        (3, EDGE_TOKENS),
        (24, list(range(100, 150))),
        (300, [i * 7919 % 200_003 for i in range(650)]),
    ]
    extra_keys = [
        ("plain", None, None),
        ("lora", ("adapter-a", "/models/adapter-a"), None),
        ("salt", None, "tenant-1"),
        # Non-ASCII LoRA strings, and a salt long enough for pickle's 4-byte
        # string length.
        ("lora_salt", ("adapter-é", "/m/é"), "s" * 300),
    ]
    for hash_algo in HASH_FNS:
        for block_size, prompt in plain_prompts:
            cases.append(
                {
                    "name": f"plain_block_size_{block_size}",
                    "hash_algo": hash_algo,
                    "block_size": block_size,
                    "num_lookahead_tokens": 0,
                    "lora": None,
                    "cache_salt": None,
                    "prompt_token_ids": prompt,
                }
            )
        for num_lookahead_tokens in (0, 1, 3):
            for name, lora, cache_salt in extra_keys:
                cases.append(
                    {
                        "name": f"{name}_lookahead_{num_lookahead_tokens}",
                        "hash_algo": hash_algo,
                        "block_size": 5,
                        "num_lookahead_tokens": num_lookahead_tokens,
                        "lora": (
                            None if lora is None else {"name": lora[0], "path": lora[1]}
                        ),
                        "cache_salt": cache_salt,
                        "prompt_token_ids": EDGE_TOKENS * 3 + [7, 8],
                    }
                )
    for name, spans in [
        ("spanning_image", [("image", "image-a", 3, 18)]),
        ("changed_image", [("image", "image-b", 3, 18)]),
        ("shifted_image", [("image", "image-a", 4, 18)]),
        ("adjacent_images", [("image", "image-a", 2, 2), ("image", "image-b", 4, 3)]),
        ("repeated_image", [("image", "image-a", 2, 2), ("image", "image-a", 12, 5)]),
        ("lookahead_image", [("image", "image-é", 9, 1)]),
        ("mixed_modalities", [("audio", "audio-a", 0, 8), ("video", "video-a", 8, 16)]),
        (
            "late_overlapping_images",
            [
                ("image", "image-a", 3, 4),
                ("image", "image-b", 18, 4),
                ("image", "image-c", 20, 4),
            ],
        ),
    ]:
        lookaheads = (
            (0, 1, 3) if name in ("spanning_image", "lookahead_image") else (0,)
        )
        for lookahead in lookaheads:
            cases.append(
                {
                    "name": f"{name}_lookahead_{lookahead}",
                    "hash_algo": "sha256_cbor",
                    "block_size": 8,
                    "num_lookahead_tokens": lookahead,
                    "frontend_fallback": name == "late_overlapping_images",
                    "lora": {"name": "adapter-é", "path": "/m/é"},
                    "cache_salt": "tenant-1",
                    "prompt_token_ids": EDGE_TOKENS * 3 + [7, 8],
                    "mm_features": [
                        {
                            "data": None,
                            "modality": modality,
                            "identifier": identifier,
                            "mm_position": {"offset": offset, "length": length},
                        }
                        for modality, identifier, offset, length in spans
                    ],
                }
            )
    cases.append(
        {
            "name": "image_negative_offset_widths",
            "hash_algo": "sha256_cbor",
            "block_size": 32,
            "num_lookahead_tokens": 0,
            "lora": None,
            "cache_salt": None,
            "prompt_token_ids": [7] * 320,
            "mm_features": [
                {
                    "data": None,
                    "modality": "image",
                    "identifier": "image-a",
                    "mm_position": {"offset": 7, "length": 300},
                }
            ],
        }
    )
    return cases


def _engine_request(case: dict[str, Any]) -> EngineCoreRequest:
    lora = case["lora"]
    sampling_params = SamplingParams(max_tokens=17)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return EngineCoreRequest(
        request_id="0",
        prompt_token_ids=case["prompt_token_ids"],
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        data_parallel_rank=None,
        lora_request=(
            None
            if lora is None
            else LoRARequest(
                lora_name=lora["name"], lora_int_id=1, lora_path=lora["path"]
            )
        ),
        cache_salt=case["cache_salt"],
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality=feature["modality"],
                identifier=feature["identifier"],
                mm_position=PlaceholderRange(**feature["mm_position"]),
            )
            for feature in case.get("mm_features", [])
        ],
    )


def _engine_block_hashes(case: dict[str, Any]) -> list[str]:
    """The prompt block hashes of the engine's `Request` for this case."""
    request = Request.from_engine_core_request(
        _engine_request(case),
        get_request_block_hasher(
            case["block_size"],
            HASH_FNS[case["hash_algo"]],
            case["num_lookahead_tokens"],
        ),
    )
    return [block_hash.hex() for block_hash in request.block_hashes]


def _engine_vectors() -> dict[str, Any]:
    """Compute every vector with the engine's default NONE_HASH seed, which is
    what an engine without PYTHONHASHSEED sends the frontend."""
    saved_seed = os.environ.pop("PYTHONHASHSEED", None)
    try:
        none_hashes: dict[str, str] = {}
        cases = []
        for hash_algo, hash_fn in HASH_FNS.items():
            init_none_hash(hash_fn)
            none_hashes[hash_algo] = kv_cache_utils.NONE_HASH.hex()
            for case in _case_inputs():
                if case["hash_algo"] == hash_algo:
                    case["block_hashes"] = _engine_block_hashes(case)
                    cases.append(case)
    finally:
        if saved_seed is not None:
            os.environ["PYTHONHASHSEED"] = saved_seed
    return {"none_hashes": none_hashes, "cases": cases}


def _format(vectors: dict[str, Any]) -> str:
    text = json.dumps(vectors, indent=2, ensure_ascii=False)
    # Keep each token id list on one line.
    text = re.sub(r"\[[\d,\s]+\]", lambda m: json.dumps(json.loads(m[0])), text)
    return text + "\n"


def test_engine_matches_frontend_prompt_block_hash_vectors():
    assert VECTORS.is_file(), f"missing {VECTORS}"
    assert json.loads(VECTORS.read_text()) == _engine_vectors(), (
        "The engine's prompt block hashes no longer match the vectors that the "
        "Rust frontend's hasher is tested against. Update "
        "rust/src/engine-core-client/src/prompt_block_hasher.rs to match, then "
        "regenerate the vectors with "
        "`.venv/bin/python tests/v1/core/test_prompt_block_hash_vectors.py`."
    )


@pytest.mark.parametrize("prefix_count", [None, 2], ids=["fallback", "partial_prefix"])
def test_engine_hashes_whole_request_when_later_media_overlap(
    monkeypatch, prefix_count
):
    """A supported prefix before an overlap does not split hashing ownership."""
    vectors = json.loads(VECTORS.read_text())
    case = next(
        case
        for case in vectors["cases"]
        if case["name"] == "late_overlapping_images_lookahead_0"
    )
    config = PrefixHashConfig(
        hash_algo=case["hash_algo"],
        hash_block_size=case["block_size"],
        none_hash=bytes.fromhex(vectors["none_hashes"][case["hash_algo"]]),
        num_lookahead_tokens=case["num_lookahead_tokens"],
    )
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    init_none_hash(HASH_FNS[case["hash_algo"]])
    incoming = _engine_request(case)
    if prefix_count is not None:
        incoming.block_hashes = [
            bytes.fromhex(value) for value in case["block_hashes"][:prefix_count]
        ]
    accepted_hashes = kv_cache_utils.frontend_block_hashes(incoming, config)
    assert accepted_hashes is None
    request = Request.from_engine_core_request(
        incoming,
        get_request_block_hasher(
            config.hash_block_size,
            HASH_FNS[config.hash_algo],
            config.num_lookahead_tokens,
        ),
        accepted_hashes,
    )
    assert [value.hex() for value in request.block_hashes] == case["block_hashes"]


if __name__ == "__main__":
    VECTORS.write_text(_format(_engine_vectors()))
    print(f"wrote {VECTORS}")
