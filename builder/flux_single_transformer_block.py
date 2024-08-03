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

model_name = "FluxSingleTransformerBlock"
ait = modeling.transformers.transformer_flux.FluxSingleTransformerBlock
config, _, pt = load_config(config_file="builder/flux_dev_config.json")

ait_module = ait(
    dim=config['num_attention_heads'] * config['attention_head_dim'],
    num_attention_heads=config['num_attention_heads'],
    attention_head_dim=config['attention_head_dim'],
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
        h_w+512,
        config['num_attention_heads'] * config['attention_head_dim'],
    ],
    name="hidden_states",
    is_input=True,
)
temb = Tensor(
    [batch, config['num_attention_heads'] * config['attention_head_dim']],
    name="temb",
    is_input=True,
)
image_rotary_emb = Tensor(
    [batch, 1, h_w+512, 64, 2, 2], # check 64
    name="image_rotary_emb",
    is_input=True,
    dtype="float32"
)


hidden_states_out = ait_module.forward(
    hidden_states=hidden_states,
    temb=temb,
    image_rotary_emb=image_rotary_emb
)
hidden_states_out = mark_output(hidden_states_out, "hidden_states_out")

target = detect_target()
compile_model(
    hidden_states_out,
    target,
    "./tmp",
    model_name,
    constants=None,
    do_constant_folding=False,
)
