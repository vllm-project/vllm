# Offline Inference with the OpenAI Batch file format

```{important}
This is a guide to performing batch inference using the OpenAI batch file format, **not** the complete Batch (REST) API.
```

## File Format

The OpenAI batch file format consists of a series of json objects on new lines.

[See here for an example file.](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl)

Each line represents a separate request. See the [OpenAI package reference](https://platform.openai.com/docs/api-reference/batch/requestInput) for more details.

```{note}
We currently support `/v1/chat/completions`, `/v1/embeddings`, and `/v1/score` endpoints (completions coming soon).
```

## Pre-requisites

* The examples in this document use `meta-llama/Meta-Llama-3-8B-Instruct`.
  - Create a [user access token](https://huggingface.co/docs/hub/en/security-tokens)
  - Install the token on your machine (Run `huggingface-cli login`).
  - Get access to the gated model by [visiting the model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and agreeing to the terms and conditions.

## Example 1: Running with a local file

### Step 1: Create your batch file

To follow along with this example, you can download the example batch, or create your own batch file in your working directory.

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl
```

Once you've created your batch file it should look like this

```bash
cat offline_inference/openai_batch/openai_example_batch.jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
```

### Step 2: Run the batch

The batch running tool is designed to be used from the command line.

You can run the batch with the following command, which will write its results to a file called `results.jsonl`

```bash
python -m vllm.entrypoints.openai.run_batch \
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

or use command-line:

```bash
vllm run-batch \
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Step 3: Check your results

You should now have your results at `results.jsonl`. You can check your results by running `cat results.jsonl`

```bash
cat results.jsonl
{"id":"vllm-383d1c59835645aeb2e07d004d62a826","custom_id":"request-1","response":{"id":"cmpl-61c020e54b964d5a98fa7527bfcdd378","object":"chat.completion","created":1715633336,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! It's great to meet you! I'm here to help with any questions or tasks you may have. What's on your mind today?"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":25,"total_tokens":56,"completion_tokens":31}},"error":null}
{"id":"vllm-42e3d09b14b04568afa3f1797751a267","custom_id":"request-2","response":{"id":"cmpl-f44d049f6b3a42d4b2d7850bb1e31bcc","object":"chat.completion","created":1715633336,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"*silence*"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":27,"total_tokens":32,"completion_tokens":5}},"error":null}
```

## Example 2: Using remote files

The batch runner supports remote input and output urls that are accessible via http/https.

For example, to run against our example input file located at `https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl`, you can run

```bash
python -m vllm.entrypoints.openai.run_batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

or use command-line:

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## Example 3: Integrating with AWS S3

To integrate with cloud blob storage, we recommend using presigned urls.

[Learn more about S3 presigned urls here]

### Additional prerequisites

* [Create an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html).
* The `awscli` package (Run `pip install awscli`) to configure your credentials and interactively use s3.
  - [Configure your credentials](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html).
* The `boto3` python package (Run `pip install boto3`) to generate presigned urls.

### Step 1: Upload your input script

To follow along with this example, you can download the example batch, or create your own batch file in your working directory.

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl
```

Once you've created your batch file it should look like this

```bash
cat offline_inference/openai_batch/openai_example_batch.jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
```

Now upload your batch file to your S3 bucket.

```bash
aws s3 cp offline_inference/openai_batch/openai_example_batch.jsonl s3://MY_BUCKET/MY_INPUT_FILE.jsonl
```

### Step 2: Generate your presigned urls

Presigned urls can only be generated via the SDK. You can run the following python script to generate your presigned urls. Be sure to replace the `MY_BUCKET`, `MY_INPUT_FILE.jsonl`, and `MY_OUTPUT_FILE.jsonl` placeholders with your bucket and file names.

(The script is adapted from <https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/s3/s3_basics/presigned_url.py>)

```python
import boto3
from botocore.exceptions import ClientError

def generate_presigned_url(s3_client, client_method, method_parameters, expires_in):
    """
    Generate a presigned Amazon S3 URL that can be used to perform an action.

    :param s3_client: A Boto3 Amazon S3 client.
    :param client_method: The name of the client method that the URL performs.
    :param method_parameters: The parameters of the specified client method.
    :param expires_in: The number of seconds the presigned URL is valid for.
    :return: The presigned URL.
    """
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod=client_method, Params=method_parameters, ExpiresIn=expires_in
        )
    except ClientError:
        raise
    return url


s3_client = boto3.client("s3")
input_url = generate_presigned_url(
    s3_client, "get_object", {"Bucket": "MY_BUCKET", "Key": "MY_INPUT_FILE.jsonl"}, 3600
)
output_url = generate_presigned_url(
    s3_client, "put_object", {"Bucket": "MY_BUCKET", "Key": "MY_OUTPUT_FILE.jsonl"}, 3600
)
print(f"{input_url=}")
print(f"{output_url=}")
```

This script should output

```text
input_url='https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091'
output_url='https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091'
```

### Step 3: Run the batch runner using your presigned urls

You can now run the batch runner, using the urls generated in the previous section.

```bash
python -m vllm.entrypoints.openai.run_batch \
    -i "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    -o "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    --model --model meta-llama/Meta-Llama-3-8B-Instruct
```

or use command-line:

```bash
vllm run-batch \
    -i "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    -o "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    --model --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Step 4: View your results

Your results are now on S3. You can view them in your terminal by running

```bash
aws s3 cp s3://MY_BUCKET/MY_OUTPUT_FILE.jsonl -
```

## Example 4: Using embeddings endpoint

### Additional prerequisites

* Ensure you are using `vllm >= 0.5.5`.

### Step 1: Create your batch file

Add embedding requests to your batch file. The following is an example:

```text
{"custom_id": "request-1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are a helpful assistant."}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are an unhelpful assistant."}}
```

You can even mix chat completion and embedding requests in the batch file, as long as the model you are using supports both chat completion and embeddings (note that all requests must use the same model).

### Step 2: Run the batch

You can run the batch using the same command as in earlier examples.

### Step 3: Check your results

You can check your results by running `cat results.jsonl`

```bash
cat results.jsonl
{"id":"vllm-db0f71f7dec244e6bce530e0b4ef908b","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-3580bf4d4ae54d52b67eee266a6eab20","body":{"id":"embd-33ac2efa7996430184461f2e38529746","object":"list","created":444647,"model":"intfloat/e5-mistral-7b-instruct","data":[{"index":0,"object":"embedding","embedding":[0.016204833984375,0.0092010498046875,0.0018358230590820312,-0.0028228759765625,0.001422882080078125,-0.0031147003173828125,...]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0}}},"error":null}
...
```

## Example 5: Using score endpoint

### Additional prerequisites

* Ensure you are using `vllm >= 0.7.0`.

### Step 1: Create your batch file

Add score requests to your batch file. The following is an example:

```text
{"custom_id": "request-1", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is the capital of France?", "text_2": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "text_1": "What is the capital of France?", "text_2": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
```

You can mix chat completion, embedding, and score requests in the batch file, as long as the model you are using supports them all (note that all requests must use the same model).

### Step 2: Run the batch

You can run the batch using the same command as in earlier examples.

### Step 3: Check your results

You can check your results by running `cat results.jsonl`

```bash
cat results.jsonl
{"id":"vllm-f87c5c4539184f618e555744a2965987","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-806ab64512e44071b37d3f7ccd291413","body":{"id":"score-4ee45236897b4d29907d49b01298cdb1","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.0010900497436523438},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
{"id":"vllm-41990c51a26d4fac8419077f12871099","custom_id":"request-2","response":{"status_code":200,"request_id":"vllm-batch-73ce66379026482699f81974e14e1e99","body":{"id":"score-13f2ffe6ba40460fbf9f7f00ad667d75","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.001094818115234375},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
```
