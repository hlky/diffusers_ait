from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.utils.import_path import import_parent

from config import load_config, mark_output


import_parent(filepath=__file__, level=1)

import modeling

batch_size = 1
resolution = 512, 1024
height, width = resolution, resolution

model_name = "FluxTransformerBlock"
ait = modeling.transformers.transformer_flux.FluxTransformerBlock
config, _, pt = load_config(config_file="builder/flux_dev_config.json")

ait_module = ait(
    dim=config["num_attention_heads"] * config["attention_head_dim"],
    num_attention_heads=config["num_attention_heads"],
    attention_head_dim=config["attention_head_dim"],
)
ait_module.name_parameter_tensor()

output_name = "Y"

vae_scale_factor = 16
seq_len = 512

batch = 1
height = IntVar(
    [height[0] // vae_scale_factor, height[1] // vae_scale_factor], "height"
)
width = IntVar([width[0] // vae_scale_factor, width[1] // vae_scale_factor], "width")
h_w = height * width
hidden_states = Tensor(
    [
        batch,
        h_w,
        config["num_attention_heads"] * config["attention_head_dim"],
    ],
    name="hidden_states",
    is_input=True,
)
encoder_hidden_states = Tensor(
    [
        batch,
        seq_len,
        config["num_attention_heads"] * config["attention_head_dim"],
    ],  # allow more tokens
    name="encoder_hidden_states",
    is_input=True,
)
temb = Tensor(
    [batch, config["num_attention_heads"] * config["attention_head_dim"]],
    name="temb",
    is_input=True,
)
image_rotary_emb = Tensor(
    [batch, 1, h_w + 512, 64, 2, 2],  # check 64
    name="image_rotary_emb",
    is_input=True,
    dtype="float32",
)


encoder_hidden_states_out, hidden_states_out = ait_module.forward(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    temb=temb,
    image_rotary_emb=image_rotary_emb,
)
encoder_hidden_states_out = mark_output(
    encoder_hidden_states_out, "encoder_hidden_states_out"
)
hidden_states_out = mark_output(hidden_states_out, "hidden_states_out")

# import torch
# from aitemplate.testing.benchmark_ait import benchmark_module
# import safetensors
# from safetensors.torch import load_file
# weights = safetensors.safe_open("G:/FLUX.1-dev/flux-dev/transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "pt")
# transformer_blocks = {}
# for key in weights.keys():
#     if not key.startswith("transformer_blocks.0."):
#         continue
#     transformer_blocks[key.replace("transformer_blocks.0.", "").replace(".", "_")] = weights.get_tensor(key).to(torch.float16)
# save_file(transformer_blocks, "H:/transformer_blocks.0.safetensors")
# constants = load_file("H:/transformer_blocks.0.safetensors", "cuda")
# module = compile_model(
# hidden_states_out,
# target,
# "./tmp",
# model_name,
# constants=constants,
# do_constant_folding=False,
# )
# benchmark_module(module, count=50, repeat=4, graph_mode=True)

target = detect_target()
module = compile_model(
    [encoder_hidden_states_out, hidden_states_out],
    target,
    "./tmp",
    model_name,
    constants=None,
    do_constant_folding=False,
)
