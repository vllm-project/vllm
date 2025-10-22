# VLLM Benchmark Profiling

This profiling directory provides a method to profile VLLM throughput and latency benchmarks using ROCm profiling utilities.

## 1. Dependencies

Before using the profiling feature, you need to install the required dependencies:

### Install ROCm Profile Data

```bash
git clone -b nvtx_enabled https://github.com/ROCm/rocmProfileData.git
cd rocmProfileData && make && sudo make install
```

### Install hipMarker

```bash
cd rocmProfileData/hipMarker && python3 setup.py install
```

## 2. Profiling Benchmarks

Profiling can be used to monitor the performance of the VLLM benchmarks with ROCm. The key flags used for profiling are:

- `--profile-rpd`: Profiles the generation process of a single batch.
- `--profile-dir PROFILE_DIR`: Specifies the path to save the profiler output, which can later be visualized using tools like [ui.perfetto.dev](https://ui.perfetto.dev/) or [chrome.tracing](chrome://tracing/).

### Profiling Using Default Directory

By default, profiling results are saved in either `vllm_benchmark_latency_result` or `vllm_benchmark_throughput_result`. To run a benchmark and profile it using the default directory, execute:

```bash
python3 benchmark_throughput.py --input-len {len} --output-len {len} --model {model} --profile-rpd
```

### Profiling With a Custom Directory

You can specify a custom directory for saving profiler outputs by using the `--profile-dir` flag:

```bash
python3 benchmark_throughput.py --input-len {len} --output-len {len} --model {model} --profile-rpd --profile-dir {/path/to/custom/dir}
```

After profiling is complete, an `.rpd` file containing the trace data will be saved to the specified directory.

## 3. Convert Trace Data to JSON Format

To view the trace data, it needs to be converted into a format that is compatible with tools like Chrome tracing or Perfetto.

You can use the `rpd2tracing.py` script in rocmProfileData to convert the `.rpd` file into a JSON file:

```bash
python3 rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
```

Once the trace is converted, open the `.json` file in [Chrome](chrome://tracing/) or [Perfetto](https://ui.perfetto.dev/) for visualization.
