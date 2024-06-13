import unittest

from typing import cast, List, Literal, Optional, Tuple

import diffusers.models.upsampling as upsampling_torch

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor

from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

import modeling.upsampling as upsampling
from utils import mark_output

# TODO: torch target detection
# TODO: cast to correct dtype

# TODO: FirUpsample2D
# TODO: KUpsample2D


class UpsamplingTestCase(unittest.TestCase):
    def _test_upsample1d(
        self,
        shape: Tuple[int, int, int],
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, seq_len = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        op = (
            upsampling_torch.Upsample1D(
                channels=channels,
                use_conv=use_conv,
                use_conv_transpose=use_conv_transpose,
            )
            .cuda()
            .half()
        )
        y_pt = op.forward(x)
        x = x.permute(0, 2, 1).contiguous()
        y = torch.empty_like(y_pt.permute(0, 2, 1).contiguous())

        X = Tensor(
            shape=[batch, seq_len, channels], dtype=dtype, name="X", is_input=True
        )
        op = upsampling.Upsample1D(
            channels=channels, use_conv=use_conv, use_conv_transpose=use_conv_transpose
        )
        Y = op.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_upsample1d_c-{channels}_conv-{use_conv}_transpose-{use_conv_transpose}",
        )

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y.permute(0, 2, 1).contiguous(),
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    # not implemented
    # def test_upsample1d(self):
    #     self._test_upsample1d(
    #         (
    #             1,
    #             16,
    #             2,
    #         ),
    #         use_conv=False,
    #         use_conv_transpose=False,
    #         tolerance=1e-4,
    #     )
    #     self._test_upsample1d(
    #         (
    #             1,
    #             16,
    #             2,
    #         ),
    #         use_conv=True,
    #         use_conv_transpose=False,
    #         tolerance=1e-4,
    #     )
    #     # self._test_upsample1d((1, 16, 2,), use_conv=False, use_conv_transpose=True, tolerance=1e-4) # not implemented

    def _test_upsample2d(
        self,
        shape: Tuple[int, int, int, int],
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        norm_type: Optional[Literal["ln_norm", "rms_norm"]] = None,
        eps: float = 1e-5,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        op = (
            upsampling_torch.Upsample2D(
                channels=channels,
                use_conv=use_conv,
                use_conv_transpose=use_conv_transpose,
                norm_type=norm_type,
                eps=eps,
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

        y_pt = op.forward(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous())

        X = Tensor(
            shape=[batch, height, width, channels], dtype=dtype, name="X", is_input=True
        )
        op = upsampling.Upsample2D(
            channels=channels,
            use_conv=use_conv,
            use_conv_transpose=use_conv_transpose,
            norm_type=norm_type,
            eps=eps,
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_upsample2d_c-{channels}_conv-{use_conv}_transpose-{use_conv_transpose}_norm-{norm_type}",
            constants=state_dict_ait,
        )

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y.permute(0, 3, 1, 2).contiguous(),
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def test_upsample2d(self):
        self._test_upsample2d(
            (
                1,
                4,
                64,
                64,
            ),
            use_conv=False,
            use_conv_transpose=False,
            tolerance=1e-4,
        )
        for norm in [None, "ln_norm", "rms_norm"]:
            self._test_upsample2d(
                (
                    1,
                    4,
                    64,
                    64,
                ),
                use_conv=True,
                use_conv_transpose=False,
                norm_type=norm,
                tolerance=1e-3,
            )
            self._test_upsample2d(
                (
                    1,
                    4,
                    64,
                    64,
                ),
                use_conv=False,
                use_conv_transpose=True,
                norm_type=norm,
                tolerance=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
