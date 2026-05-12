# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "openvla/openvla-7b"
DATASET_ID = "physical-intelligence/libero"
DATASET_SOURCE = "LeRobot/parquet conversion of OpenVLA's modified LIBERO data"
EPISODE_INDEX = 0
NUM_CASES = 10
MAX_NEW_TOKENS = 7

RUN_DIR = Path(os.environ.get("OPENVLA_RUN_DIR", "/workspace/openvla_check"))
IMAGE_DIR = RUN_DIR / "images"

CASES_PATH = RUN_DIR / "cases.json"
HF_ARTIFACTS_PATH = RUN_DIR / "hf_artifacts.pt"
VLLM_ARTIFACTS_PATH = RUN_DIR / "vllm_artifacts.pt"
RESULT_PATH = RUN_DIR / "result.json"
