import torch

from aitemplate.compiler import Model

device = "cuda"

graph_mode = False
sync = True

module = Model("")

repeat = 3
count = 23

inputs = {
    "z": torch.randn([1, 4, 64, 64], dtype=torch.float16)
    .permute(0, 2, 3, 1)
    .contiguous()
    .to(device)
}
outputs = {
    "Y": torch.empty([1, 3, 512, 512], dtype=torch.float16)
    .permute(0, 2, 3, 1)
    .contiguous()
    .to(device)
}
mean, std, _ = module.benchmark_with_tensors(
    inputs, outputs, count=count, repeat=repeat, graph_mode=graph_mode
)

inputs = {
    "z": torch.randn([1, 4, 128, 128], dtype=torch.float16)
    .permute(0, 2, 3, 1)
    .contiguous()
    .to(device)
}
outputs = {
    "Y": torch.empty([1, 3, 1024, 1024], dtype=torch.float16)
    .permute(0, 2, 3, 1)
    .contiguous()
    .to(device)
}
mean, std, _ = module.benchmark_with_tensors(
    inputs, outputs, count=count, repeat=repeat, graph_mode=graph_mode
)
