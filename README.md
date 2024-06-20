# Diffusers AIT

Diffusers for AITemplate

**Work in progress**

Depends on [`hlky/AITemplate`](https://github.com/hlky/AITemplate)

## Status

Modules in `modeling` may be initialized before porting to AIT is complete. Modules with corresponding unittests are in progress, or complete with the exception of any TODO items.

Modules that cannot currently be supported due to missing kernels in AIT, or unsupported parameters of existing kernels, are marked by raising `NotImplementedError`.

## Blockers

AIT `ops.vector_norm` aka `torch.linalg.vector_norm` only supports single reduction axes.
- `normalization`.`GlobalResponseNorm` -> 
    - `Stable Cascade`/`Wuerstchen`
    - `UVit`

Cutlass error
- `PixArtAlphaCombinedTimestepSizeEmbeddings` - `use_additional_conditions=True`
    - Possibly none - available PixArt models appear to all use `use_additional_conditions=False`

Missing `AvgPool1d`

Missing `ConvTranspose1d`

Missing `upsampling1d`

`Input tensor elementwise_15_0 not established in graph for op avg_pool2d_1`
- `ResnetBlock2D`/`ResnetBlockCondNorm2D` - `down=True`
    - TBD
```
hidden_states = x
hidden_states = self.nonlinearity(hidden_states) # elementwise_15_0
x = self.downsample(x) # seems to override `hidden_states`
hidden_states = self.downsample(hidden_states) # not established
```