from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor


def mish(x: Tensor) -> Tensor:
    X_beta = Tensor(
        shape=[],
        dtype=x.dtype(),
        name="beta",
        value=1.0,
        is_input=True,
    )
    X_threshold = Tensor(
        shape=[],
        dtype=x.dtype(),
        name="threshold",
        value=20.0,
        is_input=True,
    )
    return x * ops.tanh(ops.elementwise(ops.FuncEnum.SOFTPLUS)(x, X_beta, X_threshold))


ACTIVATION_FUNCTIONS = {
    "swish": ops.silu,
    "silu": ops.silu,
    "mish": mish,
    "gelu": ops.gelu,
    "relu": ops.relu,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias, specialization="fast_gelu")

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: Tensor) -> Tensor:
        return ops.gelu(gate)

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = ops.chunk()(hidden_states, chunks=2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x * ops.sigmoid(1.702 * x)
