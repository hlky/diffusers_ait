# Deprecated: Use [HoneyML](https://github.com/hlky/honeyml)

# Diffusers AIT

Diffusers for AITemplate

**Work in progress**

Depends on [`hlky/AITemplate`](https://github.com/hlky/AITemplate)

## Status

### Modeling

Currently base modules (activations, embeddings, resnet, etc.) are covered by unittests. Top level modules (FluxTransformer2DModel, UNet2DConditionModel, etc.) can be considered tested if a corresponding `builder` exists.

Modules that cannot currently be supported due to missing kernels in AIT, or unsupported parameters of existing kernels, are marked by raising `NotImplementedError`.

### Builders

- SD1
- SD1 Inpaint
- SD2
- SDXL
- SDXL Refiner
- SD Cascade (prior)
- SD3
- Kadinsky3
- PixArt
- Flux-dev
- Flux-schnell
- FluxTransformerBlock, FluxSingleTransformerBlock

Modules work with any supported model weights.

### Inference

Model specific inference functions are not yet implemented. For a general function see [`aitemplate.testing.benchmark_model`](https://github.com/hlky/AITemplate/blob/523529b1f8281c6f14f820e8bbf492e9fbf47c5e/python/aitemplate/testing/benchmark_ait.py#L24-L47) - this works with any module.

### Mapping

`diffusers_ait` IR is based on `Diffusers`, weight mapping should not be required for model weights in `Diffusers` format.

Mapping for other weight formats (`LDM` etc.) is required and will be implemented. See [`diffusers/scripts`](https://github.com/huggingface/diffusers/tree/main/scripts) for examples of how weights are mapped.

## Blockers

[Kernels](KERNELS.md)
