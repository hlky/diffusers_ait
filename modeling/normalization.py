from typing import Any, Dict, Iterable, Optional, Tuple, Union

from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor

from .activations import get_activation

from .embeddings import (
    CombinedTimestepLabelEmbeddings,
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


class RMSNorm(nn.Module):
    def __init__(
        self, dim, eps: float, elementwise_affine: bool = True, dtype: str = "float16"
    ):
        super().__init__()

        self.eps = eps

        if isinstance(dim, int):
            dim = [dim]

        if elementwise_affine:
            self.weight = nn.Parameter(shape=dim, value=1.0, dtype=dtype)
        else:
            self.weight = None

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype()

        hidden_states = ops.cast()(hidden_states, "float32")
        variance = ops.reduce_mean(-1, keepdim=True)(ops.pow(hidden_states, 2))
        hidden_states = hidden_states * (1.0 / ops.sqrt(variance + self.eps))

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.tensor().dtype() in ["float16", "bfloat16"]:
                hidden_states = ops.cast()(hidden_states, self.weight.tensor().dtype())
            hidden_states = hidden_states * self.weight.tensor()
        else:
            hidden_states = ops.cast()(hidden_states, input_dtype)

        return hidden_states


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int, dtype: str = "float16"):
        super().__init__()
        self.emb = nn.Embedding([num_embeddings, embedding_dim], dtype=dtype)
        self.silu = ops.silu
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2, dtype=dtype)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, dtype=dtype)

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        emb = self.linear(self.silu(self.emb(ops.flatten()(timestep))))
        scale, shift = ops.chunk()(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = ops.silu
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        class_labels: Tensor,
        hidden_dtype: Optional[Any] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        emb = self.linear(
            self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype))
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.chunk()(
            emb, chunks=6, dim=1
        )
        x = self.norm(x) * (1 + ops.unsqueeze(-1)(scale_msa)) + ops.unsqueeze(-1)(
            shift_msa
        )
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_additional_conditions: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.dtype = dtype
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
            dtype=self.dtype,
        )

        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, dtype=dtype)

    def forward(
        self,
        timestep: Tensor,
        resolution: Optional[Tensor] = None,
        aspect_ratio: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, resolution, aspect_ratio, batch_size)
        return self.linear(ops.silu(embedded_timestep)), embedded_timestep


class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self,
        embedding_dim: int,
        out_dim: int,
        num_groups: int,
        act_fn: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        channels = ops.size()(x, dim=-1)
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = ops.unsqueeze(-1)(emb)
        emb = ops.unsqueeze(-1)(emb)
        scale, shift = ops.chunk()(emb, chunks=2, dim=1)

        x = ops.group_norm(self.num_groups, channels.symbolic_value())
        x = x * (1 + scale) + shift
        return x


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
        dtype: str = "float16",
    ):
        super().__init__()
        self.silu = ops.silu
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias, dtype=dtype
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(
                embedding_dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
                bias=bias,
                dtype=dtype,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine, dtype=dtype)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: Tensor, conditioning_embedding: Tensor) -> Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = ops.chunk()(emb, 2, dim=1)
        x = self.norm(x) * ops.unsqueeze(1)(1 + scale) + ops.unsqueeze(1)(
            shift[:, None, :]
        )
        return x


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int, dtype: str = "float16"):
        super().__init__()
        self.gamma = nn.Parameter([1, 1, 1, dim], dtype=dtype)
        self.beta = nn.Parameter([1, 1, 1, dim], dtype=dtype)

    def forward(self, x: Tensor):
        gx = ops.vector_norm(dim=(1, 2), keepdim=True)(x)
        nx = gx / (ops.reduce_mean(dim=-1, keepdim=True)(gx) + 1e-6)
        return self.gamma.tensor() * (x * nx) + self.beta.tensor() + x
