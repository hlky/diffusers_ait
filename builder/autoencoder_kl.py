from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from config import load_config, mark_output

import torch


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


def map_vae(pt_module, device="cuda", dtype="float16", encoder=False):
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


batch_size = 1, 1
resolution = 8, 1024
height, width = resolution, resolution

hf_hub = "runwayml/stable-diffusion-v1-5"
model_name = f"autoencoder_kl.decoder.{resolution[1]}.{device_name}.sm{sm}"

config, ait, pt = load_config(hf_hub, subfolder="vae")

ait_module = ait(**config)
ait_module.name_parameter_tensor()

z = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        IntVar([height[0] // 8, height[1] // 8]),
        IntVar([width[0] // 8, width[1] // 8]),
        config["latent_channels"],
    ],
    name="z",
    is_input=True,
)

Y = ait_module._decode(z=z).sample
Y = mark_output(Y, "Y")

pt = pt.from_pretrained(hf_hub, subfolder="vae")
constants = map_vae(pt)

target = detect_target()

compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    constants=constants,
    dll_name=f"{model_name}.so",
)
