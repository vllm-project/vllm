# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import vllm.utils.flashinfer as flashinfer_utils


def test_has_flashinfer_cutedsl_moe_nvfp4_direct_output(monkeypatch):
    def direct_output_api(
        *,
        direct_output=False,
        output_rows_per_owner=1,
        output_physical_rows_per_owner=1,
    ):
        del direct_output
        del output_rows_per_owner
        del output_physical_rows_per_owner

    monkeypatch.setattr(
        flashinfer_utils,
        "has_flashinfer_cutedsl_moe_nvfp4",
        lambda: True,
    )
    monkeypatch.setattr(
        flashinfer_utils,
        "_get_submodule",
        lambda _: SimpleNamespace(
            cute_dsl_fused_moe_nvfp4=direct_output_api,
        ),
    )
    flashinfer_utils.has_flashinfer_cutedsl_moe_nvfp4_direct_output.cache_clear()
    assert flashinfer_utils.has_flashinfer_cutedsl_moe_nvfp4_direct_output()
    flashinfer_utils.has_flashinfer_cutedsl_moe_nvfp4_direct_output.cache_clear()


def test_rejects_flashinfer_cutedsl_moe_without_direct_output(monkeypatch):
    def materializing_api():
        pass

    monkeypatch.setattr(
        flashinfer_utils,
        "has_flashinfer_cutedsl_moe_nvfp4",
        lambda: True,
    )
    monkeypatch.setattr(
        flashinfer_utils,
        "_get_submodule",
        lambda _: SimpleNamespace(
            cute_dsl_fused_moe_nvfp4=materializing_api,
        ),
    )
    flashinfer_utils.has_flashinfer_cutedsl_moe_nvfp4_direct_output.cache_clear()
    assert not flashinfer_utils.has_flashinfer_cutedsl_moe_nvfp4_direct_output()
    flashinfer_utils.has_flashinfer_cutedsl_moe_nvfp4_direct_output.cache_clear()
