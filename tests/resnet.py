import unittest

from typing import cast, List, Optional, Tuple, Union

import diffusers.models.resnet as resnet_torch

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor

from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

import modeling.resnet as resnet
from utils import mark_output


class ResnetTestCase(unittest.TestCase):
    def _test_resnet_block_cond_norm_2d(
        self,
        shape: List[int],
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "ada_group",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        temb = get_random_torch_tensor(
            (
                [1, temb_channels, 1, 1]
                if time_embedding_norm == "spatial"
                else [1, temb_channels]
            ),
            dtype=dtype,
        )
        x_ait = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)
        temb_ait = temb.clone().to(temb.device, temb.dtype)
        if time_embedding_norm == "spatial":
            temb_ait = temb_ait.permute(0, 2, 3, 1).contiguous()

        op = (
            resnet_torch.ResnetBlockCondNorm2D(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_shortcut=conv_shortcut,
                dropout=dropout,
                temb_channels=temb_channels,
                groups=in_channels // groups,
                groups_out=groups_out,
                eps=eps,
                non_linearity=non_linearity,
                time_embedding_norm=time_embedding_norm,
                output_scale_factor=output_scale_factor,
                use_in_shortcut=use_in_shortcut,
                up=up,
                down=down,
                conv_shortcut_bias=conv_shortcut_bias,
                conv_2d_out_channels=conv_2d_out_channels,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_ait[key_ait] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Temb = Tensor(
            shape=(
                [1, 1, 1, temb_channels]
                if time_embedding_norm == "spatial"
                else [1, temb_channels]
            ),
            dtype=dtype,
            name="Temb",
            is_input=True,
        )

        op_ait = resnet.ResnetBlockCondNorm2D(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shortcut,
            dropout=dropout,
            temb_channels=temb_channels,
            groups=in_channels // groups,
            groups_out=groups_out,
            eps=eps,
            non_linearity=non_linearity,
            time_embedding_norm=time_embedding_norm,
            output_scale_factor=output_scale_factor,
            use_in_shortcut=use_in_shortcut,
            up=up,
            down=down,
            conv_shortcut_bias=conv_shortcut_bias,
            conv_2d_out_channels=conv_2d_out_channels,
            dtype=dtype,
        )
        op_ait.name_parameter_tensor()
        Y = op_ait.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_resnet_block_cond_norm_2d_{dtype}_in_channels{in_channels}"
        if time_embedding_norm is not None:
            test_name += f"_{time_embedding_norm}"
        if conv_shortcut:
            test_name += "_conv_shortcut"
        if up:
            test_name += "_up"
        if down:
            test_name += "_down"

        x = {"X": x_ait, "Temb": temb_ait}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def _test_resnet_block_2d(
        self,
        shape: List[int],
        temb_shape: Optional[List[int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        temb = (
            get_random_torch_tensor(temb_shape, dtype=dtype)
            if temb_shape is not None
            else None
        )
        x_ait = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)
        temb_ait = (
            temb.clone().to(temb.device, temb.dtype) if temb is not None else None
        )

        op = (
            resnet_torch.ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_shortcut=conv_shortcut,
                dropout=dropout,
                temb_channels=temb_channels if temb is not None else None,
                groups=groups,
                groups_out=groups_out,
                eps=eps,
                non_linearity=non_linearity,
                skip_time_act=skip_time_act,
                time_embedding_norm=time_embedding_norm,
                output_scale_factor=output_scale_factor,
                use_in_shortcut=use_in_shortcut,
                up=up,
                down=down,
                conv_shortcut_bias=conv_shortcut_bias,
                conv_2d_out_channels=conv_2d_out_channels,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_ait[key_ait] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        if temb_shape is not None:
            Temb = Tensor(
                shape=temb_shape,
                dtype=dtype,
                name="Temb",
                is_input=True,
            )
        else:
            Temb = None

        op_ait = resnet.ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shortcut,
            dropout=dropout,
            temb_channels=temb_channels if temb is not None else None,
            groups=groups,
            groups_out=groups_out,
            eps=eps,
            non_linearity=non_linearity,
            skip_time_act=skip_time_act,
            time_embedding_norm=time_embedding_norm,
            output_scale_factor=output_scale_factor,
            use_in_shortcut=use_in_shortcut,
            up=up,
            down=down,
            conv_shortcut_bias=conv_shortcut_bias,
            conv_2d_out_channels=conv_2d_out_channels,
            dtype=dtype,
        )
        op_ait.name_parameter_tensor()
        Y = op_ait.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_resnet_block_2d_{dtype}_in_channels{in_channels}"
        if time_embedding_norm is not None:
            test_name += f"_{time_embedding_norm}"
        if conv_shortcut:
            test_name += "_conv_shortcut"
        if up:
            test_name += "_up"
        if down:
            test_name += "_down"
        x = {"X": x_ait}
        if Temb is not None:
            x["Temb"] = temb_ait
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def _test_conv1d_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        n_groups: int = 8,
        activation: str = "mish",
        dtype: str = "float16",
        shape: Tuple[int, int, int] = (1, 64, 32),
        tolerance: float = 1e-5,
    ):
        batch, channels, seq_len = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        x_ait = x.clone().permute(0, 2, 1).contiguous().to(x.device, x.dtype)

        op = (
            resnet_torch.Conv1dBlock(
                inp_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                n_groups=n_groups,
                activation=activation,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key:
                value = value.permute(0, 2, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_ait[key_ait] = value

        with torch.inference_mode():
            y_pt = op.forward(x)

        y = torch.empty_like(y_pt.permute(0, 2, 1).contiguous())

        X = Tensor(
            shape=[batch, seq_len, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op_ait = resnet.Conv1dBlock(
            inp_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
            activation=activation,
            dtype=dtype,
        )
        op_ait.name_parameter_tensor()
        Y = op_ait.forward(X)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_conv1d_block_{dtype}_in_channels{in_channels}_out_channels{out_channels}"
        x = {"X": x_ait}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 2, 1).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def _test_residual_temporal_block_1d(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        activation: str = "mish",
        dtype: str = "float16",
        shape: Tuple[int, int, int] = (1, 64, 32),
        tolerance: float = 1e-5,
    ):
        batch, channels, seq_len = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        x_ait = x.clone().permute(0, 2, 1).contiguous().to(x.device, x.dtype)
        temb = get_random_torch_tensor([1, embed_dim], dtype=dtype)
        temb_ait = temb.clone()

        op = (
            resnet_torch.ResidualTemporalBlock1D(
                inp_channels=in_channels,
                out_channels=out_channels,
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                activation=activation,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key and value.ndim == 3:
                value = value.permute(0, 2, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_ait[key_ait] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 1).contiguous())

        X = Tensor(
            shape=[batch, seq_len, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Temb = Tensor(
            shape=[1, embed_dim],
            dtype=dtype,
            name="Temb",
            is_input=True,
        )

        op_ait = resnet.ResidualTemporalBlock1D(
            inp_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            activation=activation,
            dtype=dtype,
        )
        op_ait.name_parameter_tensor()
        Y = op_ait.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_residual_temporal_block_1d_{dtype}_in_channels{in_channels}_out_channels{out_channels}_dim{embed_dim}"
        x = {"X": x_ait, "Temb": temb_ait}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 2, 1).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def _test_temporal_conv_layer(
        self,
        shape: List[int],
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        dtype: str = "float16",
        num_frames: int = 1,
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_ait = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)

        op = (
            resnet_torch.TemporalConvLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout=dropout,
                norm_num_groups=norm_num_groups,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key and value.ndim == 4:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_ait[key_ait] = value

        with torch.inference_mode():
            y_pt = op.forward(x, num_frames)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op_ait = resnet.TemporalConvLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            dtype=dtype,
        )
        op_ait.name_parameter_tensor()
        Y = op_ait.forward(X, num_frames)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_temporal_conv_layer_{dtype}_in_dim{in_dim}_frames{num_frames}"
        )
        inputs_dict = {"X": x_ait}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(inputs_dict, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def test_resnet_block_cond_norm_2d(self):
        self._test_resnet_block_cond_norm_2d(
            shape=[1, 1280, 64, 64],
            in_channels=1280,
            temb_channels=1280,
            time_embedding_norm="spatial",
            tolerance=3e-3,
            dtype="float16",
        )
        self._test_resnet_block_cond_norm_2d(
            shape=[1, 1280, 64, 64],
            in_channels=1280,
            temb_channels=1280,
            tolerance=3e-3,
            up=True,
            dtype="float16",
        )
        # FIXME: Input tensor elementwise_15_0 not established in graph for op avg_pool2d_1
        # `hidden_states = x`
        # `hidden_states = self.nonlinearity(hidden_states)` # elementwise_15_0
        # `x = self.downsample(x)` # seems to override `hidden_states`
        # `hidden_states = self.downsample(hidden_states)` # not established
        # self._test_resnet_block_cond_norm_2d(
        #     shape=[1, 1280, 64, 64],
        #     in_channels=1280,
        #     temb_channels=1280,
        #     tolerance=3e-3,
        #     down=True,
        #     dtype="float16",
        # )

    def test_resnet_block_2d(self):
        self._test_resnet_block_2d(
            shape=[1, 64, 32, 32],
            temb_shape=[1, 512],
            in_channels=64,
            conv_shortcut=True,
            temb_channels=512,
            groups=32,
            groups_out=None,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=True,
            up=False,
            down=False,
            conv_shortcut_bias=True,
            conv_2d_out_channels=None,
            tolerance=2e-3,
            dtype="float16",
        )
        self._test_resnet_block_2d(
            shape=[1, 128, 32, 32],
            temb_shape=[1, 512],
            in_channels=128,
            conv_shortcut=False,
            temb_channels=512,
            groups=32,
            groups_out=None,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="scale_shift",
            output_scale_factor=1.0,
            use_in_shortcut=True,
            up=True,
            down=False,
            conv_shortcut_bias=True,
            conv_2d_out_channels=None,
            tolerance=3e-3,
            dtype="float16",
        )
        # FIXME: Input tensor elementwise_3_0 not established in graph for op avg_pool2d_0
        # self._test_resnet_block_2d(
        #     shape=[1, 64, 64, 64],
        #     temb_shape=[1, 512],
        #     in_channels=64,
        #     conv_shortcut=True,
        #     temb_channels=512,
        #     groups=32,
        #     groups_out=None,
        #     eps=1e-6,
        #     non_linearity="swish",
        #     skip_time_act=False,
        #     time_embedding_norm="default",
        #     output_scale_factor=1.0,
        #     use_in_shortcut=False,
        #     up=False,
        #     down=True,
        #     conv_shortcut_bias=True,
        #     conv_2d_out_channels=None,
        #     tolerance=3e-3,
        #     dtype="float16",
        # )
        self._test_resnet_block_2d(
            shape=[1, 64, 64, 64],
            temb_shape=None,
            in_channels=64,
            conv_shortcut=True,
            temb_channels=None,
            groups=32,
            groups_out=None,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=False,
            up=False,
            down=False,
            conv_shortcut_bias=True,
            conv_2d_out_channels=None,
            tolerance=3e-3,
            dtype="float16",
        )

    def test_conv1d_block(self):
        self._test_conv1d_block(
            in_channels=14,
            out_channels=8,
            kernel_size=5,
            dtype="float16",
            shape=(1, 14, 32),
            tolerance=3e-3,
        )

    def test_residual_temporal_block_1d(self):
        self._test_residual_temporal_block_1d(
            in_channels=14,
            out_channels=8,
            embed_dim=512,
            kernel_size=5,
            dtype="float16",
            shape=(1, 14, 32),
            tolerance=3e-3,
        )

    def test_temporal_conv_layer(self):
        self._test_temporal_conv_layer(
            shape=[1, 320, 64, 64],
            in_dim=320,
            dtype="float16",
            num_frames=1,
            tolerance=1e-3,
        )
        self._test_temporal_conv_layer(
            shape=[2, 320, 64, 64],
            in_dim=320,
            dtype="float16",
            num_frames=2,
            tolerance=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
