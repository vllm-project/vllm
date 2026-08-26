# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.sampling_params import SamplingParams


def extract_request_configs(
    sampling_params: SamplingParams | None,
    rope_profile_id: str | None = None,
) -> dict | None:
    request_configs = None
    if (
        sampling_params is not None
        and sampling_params.extra_args is not None
        and "kv_transfer_params" in sampling_params.extra_args
    ):
        kv_transfer_params = sampling_params.extra_args.get("kv_transfer_params")
        if kv_transfer_params is not None:
            assert isinstance(kv_transfer_params, dict)
            for key, value in kv_transfer_params.items():
                if key.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[key] = value
    if rope_profile_id is not None:
        if request_configs is None:
            request_configs = {}
        request_configs["lmcache.tag.rope_profile"] = rope_profile_id
    return request_configs
