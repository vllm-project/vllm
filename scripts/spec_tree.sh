# params
# model = "meta-llama/Llama-3.1-8B-Instruct"
# data = mt-bench
# num-prompts = 100
# max-num-seqs = 1
# compilation = False

# #***** vanilla *****

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 0

#***** eagle *****

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 1

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 2

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 4

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 6

# fr-spec

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 1 \
  --draft-vocab-frequency-path 'eturok/llama-3.1-8b-instruct-vocab-freq/vocab_freq.pt' \
  --draft-vocab-frequency-keep-threshold 0.25

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 2 \
  --draft-vocab-frequency-path 'eturok/llama-3.1-8b-instruct-vocab-freq/vocab_freq.pt' \
  --draft-vocab-frequency-keep-threshold 0.25

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 4 \
  --draft-vocab-frequency-path 'eturok/llama-3.1-8b-instruct-vocab-freq/vocab_freq.pt' \
  --draft-vocab-frequency-keep-threshold 0.25

VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 100 \
  --max-num-seqs 1 \
  --compilation-config '{"level": "0"}' \
  --num-spec-tokens 6 \
  --draft-vocab-frequency-path 'eturok/llama-3.1-8b-instruct-vocab-freq/vocab_freq.pt' \
  --draft-vocab-frequency-keep-threshold 0.25
