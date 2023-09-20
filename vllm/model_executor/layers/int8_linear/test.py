import torch
from w8a8linear import W8A8B8O8Linear, W8A8BFP32OFP32Linear
from icecream import ic

@torch.no_grad()
def test_w8a8b8o8_linear():
    B, M, N = 128, 512, 1024
    x = torch.randn(B, M)
    x_scale = x.abs().max() / 127
    qx = (x / x_scale).round().to(torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    y_gt = linear(x)
    y_scale = y_gt.abs().max() / 127
    q_linear = W8A8B8O8Linear.from_float(linear, x_scale, y_scale).cuda()
    q_y = q_linear(qx.cuda()).cpu()
    y_hat = q_y * y_scale
    r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
    ic(r2)

@torch.no_grad()
def test_w8a8bfp32ofp32_linear():
    B, M, N = 128, 512, 1024
    x = torch.randn(B, M)
    x_scale = x.abs().max() / 127
    qx = (x / x_scale).round().to(torch.int8)
    linear = torch.nn.Linear(M, N, bias=True)
    y_gt = linear(x)
    q_linear = W8A8BFP32OFP32Linear.from_float(linear, x_scale).cuda()
    y_hat = q_linear(qx.cuda()).cpu()
    r2 = (y_gt - y_hat).pow(2).mean() / y_gt.pow(2).mean()
    ic(r2)


if __name__ == '__main__':
    print('test_w8a8b8o8_linear')
    test_w8a8b8o8_linear()
    print('test_w8a8bfp32ofp32_linear')
    test_w8a8bfp32ofp32_linear()