## `Input shapes of the elementwise op are not compatible!`

In some cases this error is incorrect - consider the following shapes:

```
Shape1: [{'depth': 0, 'name': None, 'nop': False, 'symbolic_value': 1, 'values': [1]}, { 'depth': 0,
  'name': None,
  'nop': False,
  'symbolic_value': _sym_341,
  'values': [16, 32]}, { 'depth': 0,
  'name': None,
  'nop': False,
  'symbolic_value': _sym_342,
  'values': [16, 32]}, { 'depth': 0,
  'name': None,
  'nop': False,
  'symbolic_value': 3072,
  'values': [3072]}]

shape2: [{'depth': 0, 'name': None, 'nop': False, 'symbolic_value': 1, 'values': [1]}, { 'depth': 0,
  'name': None,
  'nop': False,
  'symbolic_value': floor(_sym_341),
  'values': [16, 32]}, { 'depth': 0,
  'name': None,
  'nop': False,
  'symbolic_value': floor(_sym_342),
  'values': [16, 32]}, { 'depth': 0,
  'name': None,
  'nop': False,
  'symbolic_value': 3072,
  'values': [3072]}]
```

Both are `[[1], [16,32], [16,32], [3072]]`. The apparent incompatibility is a subtle bug in AIT shape system.

In these cases, we simply replace the shape (or part of the shape) of one with the other.

```
out._attrs['shape'] = x._attrs['shape']
```

## Dimensions/Channels-last

AIT uses channels-last memory format. This can cause subtle bugs that can be difficult to spot.

Example from `unet_kadinsky3.py`
```
if level != 0:
    sample = ops.concatenate()([sample, hidden_states.pop()], dim=1)
```
`dim=1` here was missed on the initial port, meaning concatenate happens along `height` dimension which in turn triggers an error in `Kandinsky3ConditionalGroupNorm` because `nn.GroupNorm`'s input last dim doesn't match.

Tips and tricks will be added here.

`rank=4`/`nhwc`<->`nchw`
- If `dim=1` in PyTorch, AIT likely needs `dim=3` or `dim=-1`. `-1` is somewhat easier to use.
