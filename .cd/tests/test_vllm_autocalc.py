# SPDX-License-Identifier: Apache-2.0
import pytest
from server_autoconfig.vllm_autocalc import VarsGenerator


@pytest.fixture
def minimal_config(tmp_path):
    # Prepare minimal config files for VarsGenerator
    defaults = tmp_path / "defaults.yaml"
    varlist_conf = tmp_path / "varlist_conf.yaml"
    model_def_settings = tmp_path / "settings_vllm.csv"

    defaults.write_text("defaults:\n"
                        "  DEVICE_NAME: TEST_DEVICE\n"
                        "  HPU_MEM: {GAUDI2: 96}\n"
                        "  DTYPE: bfloat16\n")
    varlist_conf.write_text("output_vars:\n"
                            "  - DEVICE_NAME\n"
                            "  - DTYPE\n"
                            "user_variable:\n"
                            "  - DEVICE_NAME\n")
    model_def_settings.write_text("MODEL,PARAM1\nTEST_MODEL,123\n")

    return {
        "defaults_path": str(defaults),
        "varlist_conf_path": str(varlist_conf),
        "model_def_settings_path": str(model_def_settings)
    }


def test_build_context(monkeypatch, minimal_config):
    monkeypatch.setenv("MODEL", "TEST_MODEL")
    vg = VarsGenerator(**minimal_config)
    assert vg.context["DEVICE_NAME"] == "TEST_DEVICE"
    assert vg.context["DTYPE"] == "bfloat16"
    assert vg.context["MODEL"] == "TEST_MODEL"


def test_overwrite_params(monkeypatch, minimal_config):
    monkeypatch.setenv("MODEL", "TEST_MODEL")
    monkeypatch.setenv("DEVICE_NAME", "OVERRIDE_DEVICE")
    vg = VarsGenerator(**minimal_config)
    vg.overwrite_params()
    assert vg.context["DEVICE_NAME"] == "OVERRIDE_DEVICE"


def test_return_dict(monkeypatch, minimal_config):
    monkeypatch.setenv("MODEL", "TEST_MODEL")
    vg = VarsGenerator(**minimal_config)
    result = vg.return_dict()
    assert "DEVICE_NAME" in result
    assert "DTYPE" in result
    assert "MODEL" not in result  # Not in output_vars
