#!/bin/bash
# GHA uses this script to run benchmarks.

set -e
set -u

if [ $# -ne 2 ];
then
  echo "run_benchmarks needs exactly 2 arguments: "
  echo " 1. Path to a .txt file containing the list of benchmark config paths"
  echo " 2. The output path to store the benchmark results"
  exit 1
fi

benchmark_config_list_file=$1
output_directory=$2

for bench_config in `cat $benchmark_config_list_file`
do
  echo "Running benchmarks for config " $bench_config
  python3 -m neuralmagic.benchmarks.run_benchmarks -i $bench_config -o $output_directory
done
