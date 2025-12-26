#!/usr/bin/env python3
import os
import argparse

# Requires: pip install ruamel.yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
# Import CPU_Binding directly from sibling cpu_binding.py
from cpu_binding import CPU_Binding, BindingPolicy

SERVICE_NAME = "vllm-server"  # single service
XSET_NAME = "vllm_server_cpu"  # x-sets key/anchor

REQUIRED_COLUMNS = ["model_id", "input_length", "output_length", "world_size", "data_type", "num_allocated_cpu"]


def build_cpuset_and_limit(csv_path: str):
    cpus_list = ''
    idle_cpus_list = ''
    cpu_binder = CPU_Binding(csv_path=csv_path, use_hyperthread=False)
    if cpu_binder.binding_policy is BindingPolicy.Evenly_on_NUMAs or cpu_binder.cards is None:
        max_needed_numa_size = len(cpu_binder.node_to_cpus)
    elif cpu_binder.binding_policy is BindingPolicy.NUMAs_with_cards:
        max_needed_numa_size = min(cpu_binder.world_size, len(cpu_binder.node_to_cpus))
    for rank in range(max_needed_numa_size):
        rank_to_cpus = cpu_binder.get_cpus_id_binding_based_on_numa_nodes(rank)
        if rank_to_cpus not in cpus_list:
            if cpus_list != '':
                cpus_list += ','
            cpus_list += rank_to_cpus

    idle_cpus_list = ','.join(str(x) for row in cpu_binder.node_to_idle_cpus for x in row)
    print("bind cpus: ", cpus_list)
    print("idle cpus: ", idle_cpus_list)
    return cpus_list, idle_cpus_list, cpu_binder


def generate_yaml_file(cpuset_csv, num_alloc, idle_cpuset_csv, num_idle_cpus, args_cpuservice, args_output):
    yaml = YAML()
    yaml.preserve_quotes = True

    root = CommentedMap()

    services = CommentedMap()
    root["services"] = services
    vllm_server = CommentedMap()
    vllm_server["cpuset"] = DoubleQuotedScalarString(cpuset_csv)
    vllm_server["cpus"] = DoubleQuotedScalarString(str(num_alloc))

    services[SERVICE_NAME] = vllm_server

    # optional cpuservice: allocate remaining idle CPUs
    if args_cpuservice and args_cpuservice.strip():
        cpuservice = CommentedMap()
        cpuservice["cpuset"] = DoubleQuotedScalarString(idle_cpuset_csv)
        cpuservice["cpus"] = DoubleQuotedScalarString(str(num_idle_cpus))
        services[args_cpuservice.strip()] = cpuservice

    with open(args_output, "w") as f:
        yaml.dump(root, f)


def main():
    ap = argparse.ArgumentParser(description="Generate override docker-compose YAML (x-sets) for single 'vllm-server'.")
    ap.add_argument("--settings",
                    default="server/cpu_binding/cpu_binding_gnr.csv",
                    help="CSV with columns: model_id,input length,output length,world_size,num_allocated_cpu")
    ap.add_argument("--output", default="docker-compose.override.yml", help="Output compose YAML path")
    ap.add_argument("--cpuservice", help="name of the docker service binding on idle CPUs")
    args = ap.parse_args()
    model = os.environ.get("MODEL")
    if not model:
        raise RuntimeError("Set environment variable MODEL to a model_id in the CSV.")

    cpuset_csv, idle_cpuset_csv, cpu_binder = build_cpuset_and_limit(args.settings)
    num_idle_cpus = len(idle_cpuset_csv.split(","))
    generate_yaml_file(cpuset_csv, cpu_binder.num_allocated_cpu, idle_cpuset_csv, num_idle_cpus, args.cpuservice,
                       args.output)

    print(f"Wrote {args.output} for MODEL={model} ( num_allocated_cpu={cpu_binder.num_allocated_cpu})")


if __name__ == "__main__":
    main()
