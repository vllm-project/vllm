# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import os

# Third Party
import pytest
import torch
import yaml

try:
    # Third Party
    import eic

    EIC_AVAILABLE = True
except ImportError:
    EIC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not EIC_AVAILABLE, reason="eic module not available")


def parse_args():
    parser = argparse.ArgumentParser(description="EIC Storage Unit Test")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="/workspace/config/remote-eic.yaml",
        help="EIC yaml config",
    )
    args, _ = parser.parse_known_args()
    return args


def init_eic_client():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fin:
        config = yaml.safe_load(fin)

    remote_url = config.get("remote_url", None)
    if remote_url is None:
        raise AssertionError("remote_url is None")
    endpoint = remote_url[len("eic://") :]
    eic_instance_id = config.get("eic_instance_id", None)
    eic_log_dir = config.get("eic_log_dir", None)
    eic_log_level = config.get("eic_log_level", 2)
    eic_trans_type = config.get("eic_trans_type", 3)
    eic_flag_file = config.get("eic_flag_file", None)

    if not os.path.exists(eic_log_dir):
        os.makedirs(eic_log_dir, exist_ok=True)
    eic_client = eic.Client()
    init_option = eic.InitOption()
    init_option.log_dir = eic_log_dir
    init_option.log_level = eic.LogLevel(eic_log_level)
    init_option.transport_type = eic.TransportType(eic_trans_type)
    init_option.flag_file = eic_flag_file
    ret = eic_client.init(eic_instance_id, endpoint, init_option)
    if ret != 0:
        raise RuntimeError(f"EIC Client init failed with error code: {ret}")
    return eic_client


def test_set(eic_client):
    test_key = ["test_key_" + str(i) for i in range(16)]
    tensors = [
        torch.ones([12, 6, 1, 512], dtype=torch.bfloat16, device="cpu")
        for _ in range(16)
    ]
    data_keys = eic.StringVector()
    data_vals = eic.IOBuffers()
    for i in range(16):
        data_keys.append(test_key[i])
        data_vals.append(
            tensors[i].data_ptr(),
            tensors[i].numel() * tensors[i].element_size(),
            False,
        )
    set_opt = eic.SetOption()
    set_opt.ttl_second = 3
    status_code, set_outcome = eic_client.mset(data_keys, data_vals, set_opt)
    assert status_code == eic.StatusCode.SUCCESS, (
        f"Set failed with status code: {status_code}"
    )


def test_get(eic_client):
    test_key = ["test_key_" + str(i) for i in range(16)]
    tensors = [
        torch.zeros([12, 6, 1, 512], dtype=torch.bfloat16, device="cpu")
        for _ in range(16)
    ]
    data_keys = eic.StringVector()
    data_vals = eic.IOBuffers()
    for i in range(16):
        data_keys.append(test_key[i])
        data_vals.append(
            tensors[i].data_ptr(),
            tensors[i].numel() * tensors[i].element_size(),
            False,
        )
    get_opt = eic.GetOption()
    status_code, data_vals, get_outcome = eic_client.mget(data_keys, get_opt, data_vals)
    assert status_code == eic.StatusCode.SUCCESS, (
        f"Get failed with status code: {status_code}"
    )


def test_exists(eic_client):
    test_key = ["test_key_" + str(i) for i in range(16)]
    data_keys = eic.StringVector()
    for key in test_key:
        data_keys.append(key)
    exists_opt = eic.ExistOption()
    status_code, exists_outcome = eic_client.mexist(data_keys, exists_opt)
    assert status_code == eic.StatusCode.SUCCESS, (
        f"Exists failed with status code: {status_code}"
    )


def main():
    eic_client = init_eic_client()
    test_set(eic_client)
    test_exists(eic_client)
    test_get(eic_client)


if __name__ == "__main__":
    main()
