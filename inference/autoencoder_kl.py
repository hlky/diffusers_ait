from typing import List

import torch
import tqdm

from aitemplate.compiler import Model
from aitemplate.compiler.dtype import _ENUM_TO_TORCH_DTYPE

device = "cuda"

graph_mode = False
sync = True

module = Model("")
inputs = {'z': torch.randn([1, 4, 64, 64], dtype=torch.float16).permute(0, 2, 3, 1).contiguous().to(device)}
outputs = {'Y': torch.empty([1, 3, 512, 512], dtype=torch.float16).permute(0, 2, 3, 1).contiguous().to(device)}
outputs = module.run_with_tensors(inputs=inputs, outputs=outputs, sync=sync, graph_mode=graph_mode)

