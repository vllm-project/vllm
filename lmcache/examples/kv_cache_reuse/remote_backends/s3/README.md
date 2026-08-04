## LMCache can use [Amazon S3](https://aws.amazon.com/s3/) as a backend storage.

Amazon Simple Storage Service (Amazon S3) is an object storage service offering industry-leading scalability, data availability, security, and performance.

To maximize S3 performance, it's recommended to use [Amazon S3 Express One Zone storage class](https://aws.amazon.com/s3/storage-classes/express-one-zone/) and colocate your S3 bucket and [Amazon EC2 compute instance](https://aws.amazon.com/ec2/) in the same availability zone. 

## Step 1: Configure your S3 bucket and (optional) EC2 compute instance

See https://aws.amazon.com/s3/storage-classes/express-one-zone/ for configuring your S3 express-one-zone bucket. Normal S3 bucket is functional but gives worse performance.

See https://aws.amazon.com/ec2/ for configuring your own EC2 compute instance. Your own server or other cloud servers also work but give worse performance.


## Step 2: Fill out `example.yaml`

Please fill out the `BUCKET_NAME`, `AZ_ID`, and `REGION` in the `example.yaml`. 

## Step 3: Start an vLLM engine with LMCache

```bash
PYTHONHASHSEED=0 LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' --disable-log-requests --no-enable-prefix-caching
```

## Step 4: Sending requests

You should be able to see cache hit on the second time by sending the following request twice:

```bash
curl -X POST http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "'"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```