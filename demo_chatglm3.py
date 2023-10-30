import torch
from transformers import AutoConfig
from vllm.model_executor.models.modeling_chatglm import MLP, SelfAttention
from vllm.model_executor.models.chatglm3 import ChatGLM3MLP, ChatGLM3Attention


def check_mlp():
    checkpoint = "THUDM/chatglm3-6b"
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.torch_dtype = torch.float32
    print(config.torch_dtype, type(config.torch_dtype))

    device = 'cuda'

    htofh = torch.randn(config.ffn_hidden_size * 2, config.hidden_size).to(device)  # .half()
    fhtoh = torch.randn(config.hidden_size, config.ffn_hidden_size).to(device)  # .half()

    mlp = MLP(config, device=htofh.device)  # .half()
    mlp.dense_4h_to_h.weight.data.copy_(fhtoh)
    mlp.dense_h_to_4h.weight.data.copy_(htofh)
    print(mlp.dense_4h_to_h.bias, mlp.dense_h_to_4h.bias)

    mymlp = ChatGLM3MLP(config).to(device)  # .half()
    mymlp.dense_4h_to_h.weight.data.copy_(fhtoh)
    mymlp.dense_h_to_4h.weight.data.copy_(htofh)
    print(mymlp.dense_4h_to_h.bias, mymlp.dense_h_to_4h.bias)

    data = torch.randn(5, 6, config.hidden_size).to(device)  # .half()

    out1 = mlp(data)
    out2 = mymlp(data)

    print(out1[0, 0, :5])
    print(out2[0, 0, :5])

    diff = torch.max(torch.abs(out1 - out2))
    print("diff", diff)
    assert diff < 1e-6


def check_attention():
    checkpoint = "THUDM/chatglm3-6b"
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.torch_dtype = torch.float32
    device = 'cuda'

    data = torch.randn(5, 6, config.hidden_size).to(device)
    qkv_hidden_size = config.kv_channels * config.num_attention_heads + \
                      2 * config.multi_query_group_num * config.kv_channels
    qkvw = torch.randn(qkv_hidden_size, config.hidden_size).to(device)
    qkvb = torch.randn(qkv_hidden_size).to(device)
    dw = torch.randn(config.hidden_size, config.kv_channels * config.num_attention_heads).to(device)

    goldm = SelfAttention(config, layer_number=5, device=device)
    goldm.query_key_value.weight.data.copy_(qkvw)
    goldm.query_key_value.bias.data.copy_(qkvb)
    goldm.dense.weight.data.copy_(dw)
    print(goldm.dense.bias)

    myatt = ChatGLM3Attention(config).to(device)
    myatt.query_key_value.weight.data.copy_(qkvw)
    myatt.query_key_value.bias.data.copy_(qkvb)
    myatt.dense.weight.data.copy_(dw)
    print(myatt.dense.bias)

    positions = torch.repeat_interleave(torch.arange(6).reshape(-1, 6), 5, dim=0).to('cuda')
    out2 = myatt(positions, data, [None, None], None, None)

    transposed = data.permute(1, 0, 2)
    out1 = goldm(transposed).permute(1, 0, 2)

    diff = torch.max(torch.abs(out1 - out2))
    print(diff)


if __name__ == '__main__':
    # check_mlp()
    check_attention()
