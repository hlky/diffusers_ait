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
        x = get_random_torch_tensor(shape, dtype=dtype)

        op = embeddings_torch.TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=time_embed_dim
        ).to(x.device, x.dtype)

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key:
                value = value.permute(0, 2, 3, 1)
            value = value.to(x.device, x.dtype).contiguous()
            state_dict_ait[key_ait] = value

        y_pt: torch.Tensor = op.forward(x)
        y = torch.empty_like(y_pt).to(x.device, x.dtype)

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
            f"test_timestep_embedding_{dtype}_in-{in_channels}_out-{time_embed_dim}",
            constants=state_dict_ait,
        )

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y,
            y_pt.to(x.device, x.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def _test_timesteps(
        self,
        shape: List[int],
        channels: int = 320,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 0.0,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        op = embeddings_torch.Timesteps(
            num_channels=channels,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
        ).to(x.device, x.dtype)

        y_pt = op.forward(x)
        y = torch.empty_like(y_pt).to(x.device, x.dtype)

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
            "arange": torch.arange(start=0, end=channels // 2, dtype=torch.float32).to(
                x.device, x.dtype
            )
        }
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_timesteps_{dtype}_c-{channels}_flip-{flip_sin_to_cos}_scale-{downscale_freq_shift}",
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

    def _test_pixart_alpha_combined_timestep_size_embeddings(
        self,
        shape: List[int],
        embedding_dim: int = 1152,
        use_additional_conditions: bool = False,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):

        x = get_random_torch_tensor(shape, dtype=dtype)
        op = (
            embeddings_torch.PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=embedding_dim,
                size_emb_dim=embedding_dim // 3,
                use_additional_conditions=use_additional_conditions,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key:
                value = value.permute(0, 2, 3, 1)
            value = value.to(x.device, x.dtype).contiguous()
            state_dict_ait[key_ait] = value

        batch_size = x.shape[0]
        height, width = 512, 512
        if use_additional_conditions:
            resolution = (
                torch.tensor([height, width])
                .repeat(batch_size, 1)
                .to(x.device, x.dtype)
            )
            aspect_ratio = (
                torch.tensor([float(height / width)])
                .repeat(batch_size, 1)
                .to(x.device, x.dtype)
            )
        else:
            resolution = None
            aspect_ratio = None

        with torch.inference_mode():
            y_pt: torch.Tensor = op.forward(
                x, resolution, aspect_ratio, batch_size=batch_size, hidden_dtype=x.dtype
            )
        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(shape=shape, dtype=dtype, name="X", is_input=True)
        Resolution = Tensor(
            shape=[batch_size, 2], dtype=dtype, name="resolution", is_input=True
        )
        AspectRatio = Tensor(
            shape=[batch_size, 1], dtype=dtype, name="aspect_ratio", is_input=True
        )
        op = embeddings.PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim=embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X, Resolution, AspectRatio)
        Y = mark_output(Y, "Y")

        state_dict_ait.update(
            {
                "time_proj": torch.arange(
                    start=0, end=256 // 2, dtype=torch.float32
                ).to(x.device, x.dtype)
            }
        )
        if use_additional_conditions:
            state_dict_ait.update(
                {
                    "additional_condition_proj": torch.arange(
                        start=0, end=256 // 2, dtype=torch.float32
                    ).to(x.device, x.dtype),
                    "additional_condition_proj_ar": torch.arange(
                        start=0, end=256 // 2, dtype=torch.float32
                    ).to(x.device, x.dtype),
                }
            )

        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_pixart_alpha_combined_timestep_size_embeddings_{dtype}_embdim-{embedding_dim}_additional-{use_additional_conditions}",
            constants=state_dict_ait,
        )

        inputs = {"X": x}
        if use_additional_conditions:
            inputs.update({"resolution": resolution, "aspect_ratio": aspect_ratio})
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def _test_patch_embed(
        self,
        shape: List[int],
        height: int,
        width: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        interpolation_scale: float,
        flatten: bool,
        layer_norm: bool,
        bias: bool,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, h, w = shape
        latent_h, latent_w = h // 8, w // 8
        x = get_random_torch_tensor([batch, channels, latent_h, latent_w], dtype=dtype)
        x_ait = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)

        op = (
            embeddings_torch.PatchEmbed(
                height=height,
                width=width,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                interpolation_scale=interpolation_scale,
                flatten=flatten,
                layer_norm=layer_norm,
                bias=bias,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            # NOTE: Stored constants, contrary to set constants, appear to skip shape checks.
            # Correct conditions for weight modifications are important!
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_ait[key_ait] = value

        pos_embed = op.get_buffer("pos_embed").to(x.device, x.dtype)
        # NOTE: AIT is missing arange/meshgrid kernel, this must be calculated externally
        embed_h = latent_h // patch_size
        embed_w = latent_w // patch_size
        if height != embed_h or width != embed_w:
            pos_embed = embeddings.get_2d_sincos_pos_embed(
                embed_dim=pos_embed.shape[-1],
                grid_size=(embed_h, embed_w),
                base_size=op.base_size,
                interpolation_scale=interpolation_scale,
            )
            pos_embed = torch.from_numpy(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0).to(x.device, x.dtype)

        with torch.inference_mode():
            y_pt: torch.Tensor = op.forward(x)
            # NOTE: unusual case - pytorch output tensor is not contiguous, AIT requires contiguous output tensors
            # .contiguous() applied once here fixes AIT output tensor and verification
            # in practice, output from this module directly is unlikely
            y_pt = y_pt.contiguous()
        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=[batch, latent_h, latent_w, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        PosEmbed = Tensor(
            [1, (latent_h * latent_w) // 4, embed_dim],
            name="pos_embed",
            dtype=dtype,
            is_input=True,
        )
        op = embeddings.PatchEmbed(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            interpolation_scale=interpolation_scale,
            flatten=flatten,
            layer_norm=layer_norm,
            bias=bias,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X, PosEmbed)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_patch_embed_{dtype}_h{height}w{width}_patch{patch_size}_c{in_channels}_dim{embed_dim}"
        if flatten:
            test_name += "_flatten"
        if layer_norm:
            test_name += "_layer_norm"
        if bias:
            test_name += "_bias"

        inputs = {"X": x_ait, "pos_embed": pos_embed}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    # PixArtAlphaTextProjection
    def _test_pixart_alpha_text_projection(
        self,
        shape: List[int],
        caption_channels: int,
        hidden_size: int,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_ait = x.clone().to(x.device, x.dtype)

        op = (
            embeddings_torch.PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=hidden_size,
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
            y_pt: torch.Tensor = op.forward(x)
        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op = embeddings.PixArtAlphaTextProjection(
            in_features=caption_channels,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_pixart_alpha_text_projection_{dtype}_c{caption_channels}_dim{hidden_size}"
        inputs = {"X": x_ait}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
        )

    def test_timestep_embedding(self):
        self._test_timestep_embedding([1, 320], in_channels=320, tolerance=1e-3)

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

    def test_pixart_alpha_combined_timestep_size_embeddings(self):
        self._test_pixart_alpha_combined_timestep_size_embeddings(
            [1],
            embedding_dim=1152,
            use_additional_conditions=False,
            tolerance=1e-3,
        )
        self._test_pixart_alpha_combined_timestep_size_embeddings(
            [1],
            embedding_dim=1152,
            use_additional_conditions=True,
            tolerance=2e-2,  # NOTE: not good, prefer float32
            dtype="float16",
        )
        self._test_pixart_alpha_combined_timestep_size_embeddings(
            [1],
            embedding_dim=1152,
            use_additional_conditions=True,
            tolerance=1e-3,
            dtype="float32",
        )

    def test_patch_embed(self):
        self._test_patch_embed(
            shape=[1, 4, 1024, 1024],
            height=128,
            width=128,
            patch_size=2,
            in_channels=4,
            embed_dim=1152,
            interpolation_scale=2,
            flatten=True,
            layer_norm=False,
            bias=True,
            tolerance=2e-3,
            dtype="float16",
        )
        self._test_patch_embed(
            shape=[1, 4, 768, 1024],
            height=128,
            width=128,
            patch_size=2,
            in_channels=4,
            embed_dim=1152,
            interpolation_scale=2,
            flatten=True,
            layer_norm=False,
            bias=True,
            tolerance=2e-3,
            dtype="float16",
        )

    def test_pixart_alpha_text_projection(self):
        self._test_pixart_alpha_text_projection(
            shape=[1, 13, 4096],
            caption_channels=4096,
            hidden_size=1152,
            tolerance=5e-4,
            dtype="float16",
        )
        self._test_pixart_alpha_text_projection(
            shape=[1, 42, 4096],
            caption_channels=4096,
            hidden_size=1152,
            tolerance=5e-4,
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
