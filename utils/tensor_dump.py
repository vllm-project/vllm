import os
import argparse
import torch

DATA_PATH = "/root/vllm-xpu/data"


class TensorDumper:

    def __init__(self, ident: str, iter_limit: int = 3) -> None:
        self.ident = ident
        self.device = None
        self.limit = iter_limit
        self.iter_num = 0

    def append(self, data: torch.Tensor):
        if self.device is None:
            self.device = data.device.type

        if self.iter_num < self.limit:
            self.iter_num += 1
            torch.save(
                data.cpu(),
                os.path.join(
                    DATA_PATH, "{}_{}_{}".format(self.device, self.ident,
                                                 self.iter_num)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tensor dumper reader.")
    parser.add_argument("--ident", type=str)
    parser.add_argument("--iter-num", type=int)
    parser.add_argument("--full_print", action='store_true')

    args = parser.parse_args()
    ident = args.ident
    iter_num = args.iter_num
    full_print = args.full_print

    if full_print:
        torch.set_printoptions(profile=full_print)

    cuda_t = torch.load(
        os.path.join(DATA_PATH, "{}_{}_{}".format("cuda", ident, iter_num)))
    cpu_t = torch.load(
        os.path.join(DATA_PATH, "{}_{}_{}".format("cpu", ident, iter_num)))

    print("---", ident, "---", iter_num, "---")
    print("Max_diff: ", (cuda_t - cpu_t).abs().max())
    print("Mean_diff: ", (cuda_t - cpu_t).abs().mean())
    print("cuda:", cuda_t.size())
    print(cuda_t)
    print("---------------------------------")
    print("cpu:", cpu_t.size())
    print(cpu_t)
    print("---------------------------------")
