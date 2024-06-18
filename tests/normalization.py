import unittest

from typing import cast, List, Optional, Tuple

import diffusers.models.normalization as normalization_torch

import torch
from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import get_random_torch_tensor

from aitemplate.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

import modeling.normalization as normalization
from utils import mark_output


class NormalizationTestCase(unittest.TestCase):
    def _test_rms_norm(
        self,
        shape: List[int],
        hidden_size: int,
        eps: float,
        elementwise_affine: bool,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_ait = x.clone().to(x.device, x.dtype)

        op = (
            normalization_torch.RMSNorm(
                dim=hidden_size,
                eps=eps,
                elementwise_affine=elementwise_affine,
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

        op = normalization.RMSNorm(
            dim=hidden_size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_rms_norm_{dtype}_dim{hidden_size}_eps{eps}"
        if elementwise_affine:
            test_name += "_elementwise_affine"
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

    def _test_ada_layer_norm(
        self,
        shape: List[int],
        embedding_dim: int,
        num_embeddings: int,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        # NOTE: Diffusers' AdaLayerNorm expects rank 0 for timestep
        timestep = torch.randint(69, 420, [], device=x.device).to(torch.int64)
        x_ait = x.clone()
        timestep_ait = timestep.clone().unsqueeze(0)

        op = (
            normalization_torch.AdaLayerNorm(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
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
            y_pt: torch.Tensor = op.forward(x, timestep)

        y = torch.empty_like(y_pt).to(x.device, x.dtype)

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Timestep = Tensor(
            shape=[1],
            dtype="int64",
            name="timestep",
            is_input=True,
        )

        op = normalization.AdaLayerNorm(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Y = op.forward(X, Timestep)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_ada_layer_norm_{dtype}_dim{embedding_dim}_number{num_embeddings}"
        )
        inputs = {"X": x_ait, "timestep": timestep_ait}
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


    def _test_ada_layer_norm_zero(
        self,
        shape: List[int],
        hidden_size: int,
        num_embeddings: int,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, _, _ = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_ait = x.clone()
        timestep = get_random_torch_tensor([batch], dtype=dtype)
        timestep_ait = timestep.clone()
        class_labels = torch.randint(
            69, 420, [num_embeddings], device=timestep.device
        ).to(torch.int64)
        class_labels_ait = class_labels.clone().to(
            class_labels.device, class_labels.dtype
        )

        op = (
            normalization_torch.AdaLayerNormZero(
                num_embeddings=num_embeddings,
                embedding_dim=hidden_size,
            )
            .eval()
            .to(timestep.device, timestep.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_ait = {}
        for key, value in state_dict_pt.items():
            key_ait = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(timestep.device, timestep.dtype)
            state_dict_ait[key_ait] = value

        state_dict_ait.update(
            {
                "arange": torch.arange(start=0, end=256 // 2, dtype=torch.float32).to(
                    timestep.device, timestep.dtype
                )
            }
        )

        with torch.inference_mode():
            outputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = op.forward(
                x, timestep, class_labels, hidden_dtype=timestep.dtype
            )
        outputs_ait = {
            "Y": torch.empty_like(outputs[0]),
            "gate_msa": torch.empty_like(outputs[1]),
            "shift_mlp": torch.empty_like(outputs[2]),
            "scale_mlp": torch.empty_like(outputs[3]),
            "gate_mlp": torch.empty_like(outputs[4]),
        }
        for key, tensor in outputs_ait.items():
            print(f"{key} - {tensor.shape}")

        X = Tensor(
            shape=shape,
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Timestep = Tensor(
            shape=[batch],
            dtype=dtype,
            name="timestep",
            is_input=True,
        )
        ClassLabels = Tensor(
            shape=[num_embeddings],
            dtype="int64",
            name="class_labels",
            is_input=True,
        )

        op = normalization.AdaLayerNormZero(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_size,
            dtype=dtype,
        )
        op.name_parameter_tensor()
        Outputs = op.forward(X, Timestep, ClassLabels)
        Outputs = [
            mark_output(Outputs[0], "Y"),
            mark_output(Outputs[1], "gate_msa"),
            mark_output(Outputs[2], "shift_mlp"),
            mark_output(Outputs[3], "scale_mlp"),
            mark_output(Outputs[4], "gate_mlp"),
        ]

        target = detect_target()
        test_name = (
            f"test_ada_layer_norm_zero_{dtype}_num{num_embeddings}_dim{hidden_size}"
        )
        inputs = {
            "X": x_ait,
            "timestep": timestep_ait,
            "class_labels": class_labels_ait,
        }
        module = compile_model(
            Outputs,
            target,
            "./tmp",
            test_name,
            constants=state_dict_ait,
        )
        module.run_with_tensors(inputs, outputs_ait)
        for idx, name in enumerate(outputs_ait.keys()):
            y: torch.Tensor = outputs_ait[name]
            y_pt: torch.Tensor = outputs[idx]
            torch.testing.assert_close(
                y,
                y_pt.to(y.dtype),
                rtol=tolerance,
                atol=tolerance,
                msg=lambda msg: f"{msg}\n\n{name}\n\npt ({y_pt.shape}):\n{y_pt}\n\nait ({y.shape}):\n{y}\n\n",
            )


    # def test_rms_norm(self):
    #     self._test_rms_norm(
    #         shape=[1, 13, 768],
    #         hidden_size=768,
    #         eps=1e-6,
    #         elementwise_affine=True,
    #         tolerance=1e-3,
    #         dtype="float16",
    #     )
    #     self._test_rms_norm(
    #         shape=[1, 13, 768],
    #         hidden_size=768,
    #         eps=1e-6,
    #         elementwise_affine=False,
    #         tolerance=1e-3,
    #         dtype="float16",
    #     )

    def test_ada_layer_norm(self):
        self._test_ada_layer_norm(
            shape=[1, 13, 1152],
            embedding_dim=1152,
            num_embeddings=1000,
            tolerance=4e-3,
            dtype="float16",
        )
        self._test_ada_layer_norm(
            shape=[2, 13, 1152],
            embedding_dim=1152,
            num_embeddings=1000,
            tolerance=4e-3,
            dtype="float16",
        )

    def test_ada_layer_norm_zero(self):
        self._test_ada_layer_norm_zero(
            shape=[1, 1000, 1152],
            hidden_size=1152,
            num_embeddings=1000,
            tolerance=4e-3,
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
