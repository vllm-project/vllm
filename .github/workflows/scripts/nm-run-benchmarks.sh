#!/bin/bash

set -e
set -u
  
if [ $# -ne 1 ];
then
  echo "run_benchmarks needs exactly 1 argument - The output path to store the benchmark results"
  exit -1
fi
  
output_directory=$1

touch $ouptut_directory/bench_test_1.txt
touch $ouptut_directory/bench_test_2.txt