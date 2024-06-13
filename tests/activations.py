import unittest

from typing import List

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor

from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)


from diffusers.models.activations import (
    ApproximateGELU as ApproximateGELU_torch,
    GEGLU as GEGLU_torch,
    GELU as GELU_torch,
    get_activation as get_activation_torch,
)
from modeling.activations import ApproximateGELU, GEGLU, GELU, get_activation
from utils import mark_output


class ActivationsTestCase(unittest.TestCase):
    def _test_activation(
        self,
        activation_name: str,
        shape: List[int],
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        X = Tensor(shape=shape, dtype=dtype, name="X", is_input=True)
        activation = get_activation(activation_name)
        Y = activation(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"test_{activation_name}")

        x = get_random_torch_tensor(shape, dtype=dtype)
        activation_torch = get_activation_torch(activation_name)
        y_pt = activation_torch(x)
        y = torch.empty_like(y_pt)

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y,
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def test_swish(self):
        self._test_activation(
            activation_name="swish", shape=[2, 77, 768], dtype="float16", tolerance=1e-3
        )

    def test_silu(self):
        self._test_activation(
            activation_name="silu", shape=[2, 77, 768], dtype="float16", tolerance=1e-3
        )

    def test_mish(self):
        self._test_activation(
            activation_name="mish", shape=[2, 77, 768], dtype="float16", tolerance=2e-3
        )

    def test_gelu(self):
        self._test_activation(
            activation_name="gelu", shape=[2, 77, 768], dtype="float16", tolerance=1e-3
        )

    def test_relu(self):
        self._test_activation(
            activation_name="relu", shape=[2, 77, 768], dtype="float16", tolerance=1e-4
        )

    # TODO: GEGLU
    # TODO: GELU
    # TODO: ApproximateGELU


if __name__ == "__main__":
    unittest.main()
