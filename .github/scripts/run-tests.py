#!/usr/bin/env python3

import os
import re
import sys
import yaml

# # Check if filename is passed
# if len(sys.argv) != 2:
#     print("Usage: ./run-tests.py <filename>")
#     sys.exit(1)

def cc_flags(cmd, src_dir, test_dir, cc_report):
    flags = f" --cov={src_dir} --cov={test_dir} --cov-report=html:{cc_report} --cov-append "
    pytest_cmd = "pytest "
    insert_flag_pos = cmd.find(pytest_cmd) + len(pytest_cmd)
    return cmd[:insert_flag_pos] + flags + cmd[insert_flag_pos:]


def use_CUDA_VISIBLE_DEVICES(cmd):
    shard_substring = ' --shard-id=$$BUILDKITE_PARALLEL_JOB --num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT'
    shard_pos = cmd.find(shard_substring)
    cmd_to_run = cmd
    if shard_pos > 0:
        cmd_to_run = cmd.replace(shard_substring, "")
        cmd_to_run = " CUDA_VISIBLE_DEVICES=0,1 " + cmd_to_run
    return cmd_to_run


def run_cmd(cmd, test_dir):
    print(f"original: {cmd}")
    nm_cmd = cc_flags(use_CUDA_VISIBLE_DEVICES(cmd), "vllm", "tests", "cc-vllm-html")
    print(f"nm: {nm_cmd}")
    cmd_status = os.system(f"cd {test_dir} && {nm_cmd}")
    return int(cmd_status)


def run_cmds(cmds, test_dir):
    print("running commands ...")
    cmds_status = 0
    for cmd in cmds:
        cmds_status += run_cmd(cmd, test_dir)
    return cmds_status


def run_test_pipeline():

    upstream_test_cmd_file = ".buildkite/test-pipeline.yaml"

    steps = []
    with open(upstream_test_cmd_file) as stream:
        try:
            tmp = yaml.safe_load(stream)
            steps = tmp['steps']
        except Exception as ee:
            print(ee)
            raise ee

    test_status = 0
    test_dir = "tests"
    for entry in steps:
        label = entry['label']
        print(f"STARTING LABEL: {label}")

        dont_skip = set(["Regression Test", "Basic Correctness Test", "Kernels Test %N"])
        if label not in dont_skip:
            continue

        if 'command' in entry:
            cmd = entry['command']
            test_status += run_cmd(cmd, test_dir)
            print(f"completed ... {label}")
        elif 'commands' in entry:
            cmds = entry['commands']
            test_status += run_cmds(cmds, test_dir)
        else:
            print(f"step doesn't have 'command' or 'commands' ... '{entry}'")
        print(f"FINISHED LABEL: {label}")
        print()

    return test_status


if __name__ == "__main__":
    test_status = run_test_pipeline()
    print(f"test_status: {test_status}")
