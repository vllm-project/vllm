# Instructions to deploy:
1. % git clone git@github.com:hcasalet/villum.git
   % cd villum
   % docker build -f docker/Dockerfile.cpu \
  --build-arg VLLM_CPU_AVX2=true \
  --tag vllm-cpu-avx2 \
  --target vllm-openai .

2. % cd /holly
   % mkdir opt
   % sudo rsync -aP /opt/ /holly/opt/
   % sudo mv /opt /opt-bak
   % sudo ln -s /holly/opt /opt
   % sudo rm -rf /opt-bak

   % mkdir .cache
   % sudo rsync -aP $HOME/.cache/ /holly/.cache/
   % sudo mv $HOME/.cache $HOME/.cache-bak
   % sudo ln -s /holly/.cache $HOME/.cache
   % sudo rm -rf $HOME/.cache-back

   % sudo mkdir -p /opt/vllm_kvshare
   % mkdir -p $HOME/.cache/huggingface

3. % cd /holly/villum/pd-compose
   % export HF_TOKEN=""
   % docker compose up

4. Verifications;
   % curl -sSf http://localhost:8100/v1/models | jq .
   % curl -sSf http://localhost:8200/v1/models | jq .

   % curl -sSf http://localhost:8100/v1/completions \
     -H "Content-Type: application/json" \
     -d '{ \
        "model": "facebook/opt-125m", \
        "prompt": "Hello from prefill. Continue:", \
        "max_tokens": 1, \
        "temperature": 0 \
    }' | jq .
    % curl -sSf http://localhost:8200/v1/completions \
      -H "Content-Type: application/json" \
      -d '{ \
        "model": "facebook/opt-125m", \
        "prompt": "Hello from decode. Continue:", \
        "max_tokens": 16, \
        "temperature": 0 \
    }' | jq .

    % curl -sSf http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "facebook/opt-125m", \
            "prompt": "Disaggregated prefill/decode test. Finish this sentence: CPU-only vLLM is", \
            "max_tokens": 32, \
            "temperature": 0 \
        }' | jq .