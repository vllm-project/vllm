from vllm import LLM
from vllm.assets.image import ImageAsset

image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
prompt = "<|image_1|> Represent the given image with the following question: What is in the image"  # noqa: E501

# Create an LLM.
llm = LLM(
    model="TIGER-Lab/VLM2Vec-Full",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    mm_processor_kwargs={"num_crops": 16},
)

# Generate embedding. The output is a list of EmbeddingRequestOutputs.
outputs = llm.encode({"prompt": prompt, "multi_modal_data": {"image": image}})

# Print the outputs.
for output in outputs:
    print(output.outputs.embedding)  # list of ??? floats
