import json

from typing import Optional

import diffusers.models.transformers
import diffusers.models.unets
import requests

from aitemplate.utils.import_path import import_parent

from diffusers.models.model_loading_utils import _CLASS_REMAPPING_DICT

import_parent(filepath=__file__, level=1)

import modeling

from utils import mark_output

_CLASS_MAPPING = {
    "DiTTransformer2DModel": {
        "ait": modeling.transformers.DiTTransformer2DModel,
        "pt": diffusers.models.transformers.DiTTransformer2DModel,
    },
    "DualTransformer2DModel": {
        "ait": modeling.transformers.DualTransformer2DModel,
        "pt": diffusers.models.transformers.DualTransformer2DModel,
    },
    "HunyuanDiT2DModel": {
        "ait": modeling.transformers.HunyuanDiT2DModel,
        "pt": diffusers.models.transformers.HunyuanDiT2DModel,
    },
    "PixArtTransformer2DModel": {
        "ait": modeling.transformers.PixArtTransformer2DModel,
        "pt": diffusers.models.transformers.PixArtTransformer2DModel,
    },
    "PriorTransformer": {
        "ait": modeling.transformers.PriorTransformer,
        "pt": diffusers.models.transformers.PriorTransformer,
    },
    "Transformer2DModel": {
        "ait": modeling.transformers.Transformer2DModel,
        "pt": diffusers.models.transformers.Transformer2DModel,
    },
    "SD3Transformer2DModel": {
        "ait": modeling.transformers.SD3Transformer2DModel,
        "pt": diffusers.models.transformers.SD3Transformer2DModel,
    },
    "TransformerTemporalModel": {
        "ait": modeling.transformers.TransformerTemporalModel,
        "pt": diffusers.models.transformers.TransformerTemporalModel,
    },
    "UNet1DModel": {
        "ait": modeling.unets.UNet1DModel,
        "pt": diffusers.models.unets.UNet1DModel,
    },
    "UNet2DModel": {
        "ait": modeling.unets.UNet2DModel,
        "pt": diffusers.models.unets.UNet2DModel,
    },
    "UNet2DConditionModel": {
        "ait": modeling.unets.UNet2DConditionModel,
        "pt": diffusers.models.unets.UNet2DConditionModel,
    },
    "UNet3DConditionModel": {
        "ait": modeling.unets.UNet3DConditionModel,
        "pt": diffusers.models.unets.UNet3DConditionModel,
    },
    "I2VGenXLUNet": {
        "ait": modeling.unets.I2VGenXLUNet,
        "pt": diffusers.models.unets.I2VGenXLUNet,
    },
    "Kandinsky3UNet": {
        "ait": modeling.unets.Kandinsky3UNet,
        "pt": diffusers.models.unets.Kandinsky3UNet,
    },
    "UNetMotionModel": {
        "ait": modeling.unets.UNetMotionModel,
        "pt": diffusers.models.unets.UNetMotionModel,
    },
    "UNetSpatioTemporalConditionModel": {
        "ait": modeling.unets.UNetSpatioTemporalConditionModel,
        "pt": diffusers.models.unets.UNetSpatioTemporalConditionModel,
    },
    "StableCascadeUNet": {
        "ait": modeling.unets.StableCascadeUNet,
        "pt": diffusers.models.unets.StableCascadeUNet,
    },
    "UVit2DModel": {
        "ait": modeling.unets.UVit2DModel,
        "pt": diffusers.models.unets.UVit2DModel,
    },
}


def load_config(hf_hub: str, subfolder: Optional[str] = None):
    filename = "config.json"
    if subfolder:
        filename = f"{subfolder}/{filename}"
    url = f"https://huggingface.co/{hf_hub}/resolve/main/{filename}?download=true"
    r = requests.get(url)
    if not r.ok:
        return
    try:
        j = r.json()
    except Exception as e:
        print(e)
    config = j
    _class_name = config.get("_class_name", "")
    _diffusers_version = config.pop("_diffusers_version")
    remapped_class = _CLASS_REMAPPING_DICT.get(_class_name).get(
        config["norm_type"], None
    )
    if remapped_class:
        _class_name = remapped_class
        config["_class_name"] = _class_name
    print(_class_name)
    classes = _CLASS_MAPPING.get(_class_name, None)
    if classes:
        return config, classes["ait"], classes["pt"]
    return None, None, None
