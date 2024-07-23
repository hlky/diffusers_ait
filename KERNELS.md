# Kernels

Missing or incomplete AIT kernels are documented here, optionally detailing which models are blocked.

### ops.vector_norm
aka `torch.linalg.vector_norm` only supports single reduction axes.
- `normalization`.`GlobalResponseNorm` -> 
    - `Stable Cascade`/`Wuerstchen`
    - `UVit`

### ops.reduce_*
only supports single reduction axes.

### meshgrid
Not required but nice to have. Would allow `PosEmbed` (various Transformer2dModel) to be calculated internally rather than provided as an additional input.

### Other
#### Cutlass error
- `PixArtAlphaCombinedTimestepSizeEmbeddings` - `use_additional_conditions=True`
    - Possibly none - available PixArt models appear to all use `use_additional_conditions=False`

#### `Input tensor elementwise_15_0 not established in graph for op avg_pool2d_1`
- `ResnetBlock2D`/`ResnetBlockCondNorm2D` - `down=True`
    - TBD
```
hidden_states = x
hidden_states = self.nonlinearity(hidden_states) # elementwise_15_0
x = self.downsample(x) # seems to override `hidden_states`
hidden_states = self.downsample(hidden_states) # not established
```