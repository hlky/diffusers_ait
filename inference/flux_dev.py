import torch
import tqdm
from typing import List

from aitemplate.compiler import Model
from aitemplate.compiler.dtype import _ENUM_TO_TORCH_DTYPE

module_out_path = "H:/ait_modules/flux-dev"
device = "cuda"


def ProcessBlock(block_name, total_blocks, modules_to_load=4):
    inputs = {}
    outputs = {}
    block_idx = 0
    loaded_idx = 0
    pbar = tqdm.tqdm(range(0, total_blocks), desc=block_name)
    modules: List[Model] = []
    for idx in range(0, modules_to_load):
        pbar.set_postfix_str(f"loading {block_name}.{idx}.dll")
        module = Model(f"{module_out_path}/{block_name}.{idx}.dll")
        modules.append(module)
        loaded_idx += 1
    while block_idx < total_blocks:
        module = modules.pop(0)
        if block_idx == 0:
            for name, idx in module.get_input_name_to_index_map().items():
                shape = module.get_input_maximum_shape(idx)
                dtype = _ENUM_TO_TORCH_DTYPE[module.get_input_dtype(idx)]
                tensor = torch.randn(*shape, dtype=dtype).to(device)
                inputs[name] = tensor
        else:
            inputs["hidden_states"] = outputs["hidden_states_out"]
            if "encoder_hidden_states" in inputs:
                inputs["encoder_hidden_states"] = outputs["encoder_hidden_states_out"]
        if block_idx == 0:
            for name, idx in module.get_output_name_to_index_map().items():
                shape = module.get_output_maximum_shape(idx)
                dtype = _ENUM_TO_TORCH_DTYPE[module.get_output_dtype(idx)]
                tensor = torch.empty(*shape, dtype=dtype).to(device)
                outputs[name] = tensor
        module.run_with_tensors(inputs=inputs, outputs=outputs)
        del module
        pbar.update()
        if loaded_idx < total_blocks:
            pbar.set_postfix_str(f"loading {block_name}.{loaded_idx}.dll")
            modules.append(Model(f"{module_out_path}/{block_name}.{loaded_idx}.dll"))
            loaded_idx += 1
        block_idx += 1
    return outputs


for _ in tqdm.tqdm(range(20), desc="steps"):
    _ = ProcessBlock("FluxTransformerBlock", 19, modules_to_load=1)
    _ = ProcessBlock("FluxSingleTransformerBlock", 38, modules_to_load=1)
