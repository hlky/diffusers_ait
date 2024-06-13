import unittest

from typing import cast, List, Optional

import diffusers.models.embeddings as embeddings_torch

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor

from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

import modeling.embeddings as embeddings
from utils import mark_output

# TODO: other embeddings


class EmbeddingsTestCase(unittest.TestCase):
    def _test_timestep_embedding(
        self,
        shape: List[int],
        in_channels: int = 320,
        time_embed_dim: Optional[int] = None,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        if time_embed_dim is None:
            time_embed_dim = in_channels * 4

        op = (
            embeddings_torch.TimestepEmbedding(
                in_channels=in_channels, time_embed_dim=time_embed_dim
            )
            .cuda()
            .half()
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key:
                value = value.permute(0, 2, 3, 1)
            value = value.half().cuda().contiguous()
            state_dict_ait[key_ait] = value

        x = get_random_torch_tensor(shape, dtype=dtype)

        y_pt = op.forward(x)
        y = torch.empty_like(y_pt)

        X = Tensor(shape=shape, dtype=dtype, name="X", is_input=True)
        op = embeddings.TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=time_embed_dim, dtype=dtype
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_timestep_embedding_in-{in_channels}_out-{time_embed_dim}",
            constants=state_dict_ait,
        )

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y,
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def test_timestep_embedding(self):
        self._test_timestep_embedding([1, 320], in_channels=320, tolerance=1e-3)

    def _test_timesteps(
        self,
        shape: List[int],
        channels: int = 320,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 0.0,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        op = (
            embeddings_torch.Timesteps(
                num_channels=channels,
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=downscale_freq_shift,
            )
            .cuda()
            .half()
        )

        x = get_random_torch_tensor(shape, dtype=dtype)

        y_pt = op.forward(x)
        y = torch.empty_like(y_pt, dtype=torch.float16)

        X = Tensor(shape=shape, dtype=dtype, name="X", is_input=True)
        op = embeddings.Timesteps(
            num_channels=channels,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")

        constants = {
            "arange": torch.arange(start=0, end=channels // 2, dtype=torch.float32)
            .cuda()
            .half()
        }
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_timesteps_c-{channels}_flip-{flip_sin_to_cos}_scale-{downscale_freq_shift}",
            constants=constants,
        )

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def test_timesteps(self):
        self._test_timesteps(
            [1],
            channels=320,
            flip_sin_to_cos=False,
            downscale_freq_shift=0.0,
            tolerance=1e-3,
        )
        self._test_timesteps(
            [1],
            channels=320,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            tolerance=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
