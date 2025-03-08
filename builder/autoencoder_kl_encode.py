from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from config import load_config, mark_output

import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--max", type=int)
parser.add_argument("--min", type=int, default=8)
parser.add_argument("--min-batch", type=int, default=1)
parser.add_argument("--max-batch", type=int, default=1)
parser.add_argument("--hf_hub", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--label", type=str, default="v1")
parser.add_argument("--subfolder", type=str, default=None, help="`vae` if `hf_hub` is for a pipeline.")
parser.add_argument("--dtype", type=str, default="float16")

args = parser.parse_args()


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


def map_vae(pt_module, device="cuda", dtype="float16", encoder=True):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    params_ait = {}
    quant_key = "post_quant" if encoder else "quant"
    vae_key = "decoder" if encoder else "encoder"
    for key, arr in pt_params.items():
        if key.startswith(vae_key):
            continue
        if key.startswith(quant_key):
            continue
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        key = key.replace(".", "_")
        if (
            "conv" in key
            and "norm" not in key
            and key.endswith("_weight")
            and len(arr.shape) == 4
        ):
            params_ait[key] = torch.permute(arr, [0, 2, 3, 1]).contiguous()
        else:
            params_ait[key] = arr
    if encoder:
        params_ait["encoder_conv_in_weight"] = torch.functional.F.pad(
            params_ait["encoder_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
        )

    return params_ait


device_name = (
    torch.cuda.get_device_name()
    .lower()
    .replace("nvidia ", "")
    .replace("geforce rtx ", "")
    .replace("geforce gtx ", "")
    .replace("geforce gt ", "")
    .replace("geforce ", "")
    .replace("tesla ", "")
    .replace("quadro ", "")
    .strip()
    .replace(" ", "_")
    .lower()
    .split(",")[0]
    .split("(")[0]
)

sm = "".join(str(i) for i in torch.cuda.get_device_capability())


batch_size = args.min_batch, args.max_batch
resolution = args.min, args.max
height, width = resolution, resolution

hf_hub = args.hf_hub
label = args.label
model_name = f"autoencoder_kl.encoder.{label}.{resolution[1]}.{device_name}.sm{sm}"

config, ait, pt = load_config(hf_hub, subfolder=args.subfolder)

ait_module = ait(**config)
ait_module.name_parameter_tensor()

x = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        IntVar([height[0], height[1]]),
        IntVar([width[0], width[1]]),
        config["out_channels"],
    ],
    name="x",
    is_input=True,
    dtype=args.dtype,
)
sample = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        IntVar([height[0] // 8, height[1] // 8]),
        IntVar([width[0] // 8, width[1] // 8]),
        config["out_channels"],
    ],
    name="sample",
    is_input=True,
    dtype=args.dtype,
)

latents = ait_module.encode(x=x).latent_dist.sample(sample)
latents = mark_output(latents, "latents")

pt = pt.from_pretrained(hf_hub, subfolder=args.subfolder)
constants = map_vae(pt)

target = detect_target()

compile_model(
    latents,
    target,
    "./tmp",
    model_name,
    constants=constants,
    dll_name=f"{model_name}.so",
)
