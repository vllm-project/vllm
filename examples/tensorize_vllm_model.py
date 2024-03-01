from vllm import LLM
from tensorizer import TensorSerializer

model_ref = "EleutherAI/gpt-j-6B"
# For less intensive requirements, swap above with the line below:
# model_ref = "EleutherAI/gpt-neo-125M"
model_name = model_ref.split("/")[-1]
# Change this to your S3 bucket.
s3_bucket = "bucket"
s3_uri = f"s3://{s3_bucket}/{model_name}.tensors"

model = LLM(model=model_ref)

serializer = TensorSerializer(s3_uri)
serializer.write_module(model)
serializer.close()