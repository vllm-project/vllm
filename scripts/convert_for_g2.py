import torch
from safetensors import safe_open
from safetensors.torch import save_file

path="/data/DeepSeek-R1/model-00"

model_tail="-of-000163.safetensors"

out_path="/data/DeepSeek-R1-G2/model-00"

for i in range(163):
    idx = str(i + 1).zfill(3)
    model_path = path + idx + model_tail
    print("Path = " + str(model_path) + " conversion...")
    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "proj" in k and "scale_inv" in k:
                result = f.get_tensor(k) * 448.0 / 240.0
            elif "proj" in k and not ("scale_inv" in k) and not ("eh_" in k):
                result = (f.get_tensor(k).float() * 240.0 / 448.0).to(torch.float8_e4m3fn)
            else:
                result = f.get_tensor(k)
            tensors.update({k : result})

    output_path = out_path + idx + model_tail
    save_file(tensors, output_path)

