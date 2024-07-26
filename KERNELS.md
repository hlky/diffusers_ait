# Kernels

Missing or incomplete AIT kernels are documented here, optionally detailing which models are blocked.

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