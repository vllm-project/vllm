
echo "Setting up proxy configuration..."
export HTTPS_PROXY=http://fwdproxy:8080
export HTTP_PROXY=http://fwdproxy:8080
export FTP_PROXY=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export http_proxy=http://fwdproxy:8080
export ftp_proxy=http://fwdproxy:8080
export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb,127.0.0.1,localhost"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# for Triton
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=TRITON_MLA
export VLLM_LOG_LEVEL=DEBUG
pytest tests/distributed/test_context_parallel.py -s
