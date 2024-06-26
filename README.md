# Diffusers AIT

Diffusers for AITemplate

**Work in progress**

Depends on [`hlky/AITemplate`](https://github.com/hlky/AITemplate)

## Status

### Modeling

Modules in `modeling` may be initialized before porting to AIT is complete. Modules with corresponding unittests are in progress, or complete with the exception of any TODO items.

Modules that cannot currently be supported due to missing kernels in AIT, or unsupported parameters of existing kernels, are marked by raising `NotImplementedError`.

## Blockers

[Kernels](KERNELS.md)
