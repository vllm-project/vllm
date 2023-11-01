import sys

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as VRotaryEmbedding


def check_rope():
    head_size = 64
    max_seq_len = 128
    base = 10000
    vllm_rot = VRotaryEmbedding(
        head_size=head_size, rotary_dim=head_size, max_position_embeddings=max_seq_len,
        base=base, is_neox_style=False, is_glm_style=True)

    shape = (5, 6, 7, head_size)  # b,s,t,h
    query = torch.randn(*shape)
    cq = torch.clone(query)
    key = torch.randn(*shape)

    positions = torch.repeat_interleave(torch.arange(6).reshape(-1, 6), 5, dim=0).to('cuda')
    query = query.reshape(5, 6, -1).to('cuda')
    key = key.reshape(5, 6, -1).to('cuda')

    query, key = vllm_rot(positions, query, key)
    query = query.cpu()
    out1 = query.reshape(5, 6, 7, -1)

    re = RotaryEmbedding(dim=head_size // 2)
    cos_sin = re(max_seq_len)
    out = apply_rotary_pos_emb(cq.permute(1, 0, 2, 3), cos_sin)
    out2 = out.permute(1, 0, 2, 3)

    print("diff", torch.max(torch.abs(out1 - out2)))


def run_llm():
    from vllm import LLM, SamplingParams
    llm = LLM(model="THUDM/chatglm3-6b", trust_remote_code=True, dtype='float32')
    # print(len(llm.llm_engine.workers), llm.llm_engine.workers[0].model)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "希望这篇文章能",
        "给六岁小朋友解释一下万有引",
    ]
    # sampling_params = SamplingParams(temperature=0.1, top_p=0.95)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16)
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # print(output)


def run_transformer():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, torch_dtype=torch.float32).cuda()
    model = model.eval()
    tko = tokenizer("给六岁小朋友解释一下万有引")
    input_ids = torch.LongTensor([tko.input_ids]).to('cuda')
    print("00000", input_ids)

    outputs = model(input_ids, output_hidden_states=True)

    for i in range(len(outputs.hidden_states)):
        hidden_state = outputs.hidden_states[i]
        hidden_state = hidden_state.permute(1, 0, 2)
        print(f"gold {i}", hidden_state[0, 3, :5])
    # vhs = torch.load(path)
    # print(vhs.size())
    # print(vhs[0, 0, :5])
    # print(torch.max(torch.abs(hidden_state - vhs)))
    # print(torch.min(torch.abs(hidden_state - vhs)))

    outputs = torch.argmax(outputs.logits, dim=-1).tolist()[0]
    print(outputs)
    response = tokenizer.decode(outputs)
    # for token in response:
    #     print(token)


if __name__ == '__main__':
    # check_rope()
    run_llm()
    # run_transformer()
