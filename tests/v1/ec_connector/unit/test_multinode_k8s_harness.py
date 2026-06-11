# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "cpu_ec_connector"
sys.path.insert(0, str(SCRIPTS_DIR))

K8S_DIR = SCRIPTS_DIR / "k8s"

_PATCH_KWARGS = dict(
    run_id="testrun",
    namespace="test-ns",
    image="my-reg/vllm:test",
    model="TestModel",
    port=8001,
    gpu_memory_utilization=0.05,
    ec_role="ec_producer",
    engine_id="ec-producer-0",
    num_ec_blocks=80000,
    side_channel_port=5601,
    producer=True,
    different_nodes=False,
)


def _env(container, name):
    return next(
        (e.get("value") for e in container.get("env", []) if e["name"] == name),
        None,
    )


@pytest.fixture
def producer_patched():
    from k8s_harness import patch_deployment_yaml

    return patch_deployment_yaml(K8S_DIR / "producer-deployment.yaml", **_PATCH_KWARGS)


def test_patch_name_and_namespace(producer_patched):
    assert producer_patched["metadata"]["name"] == "vllm-ec-producer-testrun"
    assert producer_patched["metadata"]["namespace"] == "test-ns"


def test_patch_image(producer_patched):
    container = producer_patched["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "my-reg/vllm:test"


def test_patch_model_env(producer_patched):
    container = producer_patched["spec"]["template"]["spec"]["containers"][0]
    assert _env(container, "VLLM_MODEL") == "TestModel"


def test_patch_port_env(producer_patched):
    container = producer_patched["spec"]["template"]["spec"]["containers"][0]
    assert _env(container, "VLLM_PORT") == "8001"


def test_patch_gpu_mem_env(producer_patched):
    container = producer_patched["spec"]["template"]["spec"]["containers"][0]
    assert _env(container, "GPU_MEMORY_UTILIZATION") == "0.05"


def test_patch_ec_config_env(producer_patched):
    container = producer_patched["spec"]["template"]["spec"]["containers"][0]
    cfg = json.loads(_env(container, "EC_TRANSFER_CONFIG"))
    assert cfg["ec_role"] == "ec_producer"
    assert cfg["ec_connector_extra_config"]["num_ec_blocks"] == 80000


def test_patch_configmap_volume_name(producer_patched):
    volumes = producer_patched["spec"]["template"]["spec"]["volumes"]
    cm_vol = next(v for v in volumes if v["name"] == "sitecustomize")
    assert cm_vol["configMap"]["name"] == "ec-test-sitecustomize-testrun"


def test_no_anti_affinity_by_default(producer_patched):
    affinity = producer_patched["spec"]["template"]["spec"].get("affinity", {})
    assert "podAntiAffinity" not in affinity


def test_anti_affinity_when_different_nodes():
    from k8s_harness import patch_deployment_yaml

    result = patch_deployment_yaml(
        K8S_DIR / "producer-deployment.yaml",
        **{**_PATCH_KWARGS, "different_nodes": True},
    )
    affinity = result["spec"]["template"]["spec"].get("affinity", {})
    assert "podAntiAffinity" in affinity


def test_consumer_deployment_patches():
    from k8s_harness import patch_deployment_yaml

    result = patch_deployment_yaml(
        K8S_DIR / "consumer-deployment.yaml",
        **{
            **_PATCH_KWARGS,
            "port": 8002,
            "ec_role": "ec_consumer",
            "engine_id": "ec-consumer-0",
            "producer": False,
        },
    )
    assert result["metadata"]["name"] == "vllm-ec-consumer-testrun"
    container = result["spec"]["template"]["spec"]["containers"][0]
    assert _env(container, "VLLM_PORT") == "8002"
    cfg = json.loads(_env(container, "EC_TRANSFER_CONFIG"))
    assert cfg["ec_role"] == "ec_consumer"
