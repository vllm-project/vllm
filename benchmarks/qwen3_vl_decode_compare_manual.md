# Qwen3-VL video decode comparison

Compare full-resolution OpenCV CPU and TorchCodec CUDA decoding through Qwen3VLVideoBackend.load_bytes.
Run each backend in a separate process with the same 1080p video. The default is one warmup and five measured runs.
The CUDA path requires TorchCodec and an NVDEC-capable GPU; the script rejects CPU fallback.

From the repository root, run the CPU case and then pass its JSON as --reference to the CUDA case.
Use --help to see the required video path, --backend, --reference and --output arguments.
The reference checks input SHA256, sampled indices and output shape; it does not check pixel identity.
Timing includes decoder construction and frame retrieval, but excludes file reading and model preprocessing.
Memory peaks are sampled every 5 ms and may miss short peaks.

Example commands from the repository root:
    PYTHONPATH="$PWD" python benchmarks/qwen3_vl_decode_compare_manual.py /path/to/1080p.mp4 --backend opencv --output /tmp/opencv.json
    PYTHONPATH="$PWD" python benchmarks/qwen3_vl_decode_compare_manual.py /path/to/1080p.mp4 --backend torchcodec --reference /tmp/opencv.json --output /tmp/torchcodec.json

The JSON records the source commit. The installed vLLM package version can differ from the source checkout.
