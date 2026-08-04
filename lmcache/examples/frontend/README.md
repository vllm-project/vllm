# Online chat with frontend and context
This will help you set up vLLM + LMCache and a QA frontend.  
The default context is a ffmpeg man page.  
## Prerequisites
Your server should have at least 1 GPU.  

This will use the port 8000 (for vLLM), 8501 (for the frontend) and port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. In one terminal,  
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'```  
Wait until it's ready.  
3. Launch frontend.  
```pip install openai streamlit```  
```streamlit run frontend.py```  
Then open that URL of Streamlit app in browser.  
## What to expect
LMCache should be able to reduce the response delay since the second question.  