#!/usr/bin/env python3

import os
import sys
import yaml

# # Check if filename is passed
# if len(sys.argv) != 2:
#     print("Usage: ./run-tests.py <filename>")
#     sys.exit(1)

def run_cmd(cmd):
    print(f"{cmd}")


def run_cmds(cmds):
    print("running commands ...")
    for cmd in cmds:
        run_cmd(cmd)


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

    for entry in steps:
        label = entry['label']
        print(f"STARTING LABEL: {label}")
        if 'command' in entry:
            cmd = entry['command']
            run_cmd(cmd)
            print(f"completed ... {label}")
        elif 'commands' in entry:
            cmds = entry['commands']
            run_cmds(cmds)
        else:
            print(f"step doesn't have 'command' or 'commands' ... '{entry}'")
        print(f"FINISHED LABEL: {label}")
        print()


if __name__ == "__main__":
    run_test_pipeline()
