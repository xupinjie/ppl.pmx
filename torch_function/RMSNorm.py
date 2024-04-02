import torch
from pathlib import Path
import os

import sys
sys.path.append("../")  # 添加上一级目录到sys.path
from model_zoo import ModelUtils
TensorDumper = ModelUtils.__TensorDumperV2__()

class RMSNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value,
        axis: int = -1, eps: float = 1e-5):
        Y = g.op("pmx::RMSNorm", X, weight,
                    axis_i = axis, eps_f = eps, skip_term_i = False)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor,
        axis: int = -1, eps: float = 1e-5):
        if torch.onnx.is_in_onnx_export():
            return X
        x = X.float()
        mean_square = (x * x).mean(axis, keepdim=True)
        Y = x * torch.rsqrt(mean_square + eps)
        return Y.type_as(X) * weight


class SkipRMSNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value, SkipIn: torch.Value = None,
        axis: int = -1, eps: float = 1e-5):
        if SkipIn is None:
            Y, SkipOut = g.op("pmx::RMSNorm", X, weight,
                        axis_i = axis, eps_f = eps, skip_term_i = True,
                        outputs = 2)
        else:
            Y, SkipOut = g.op("pmx::RMSNorm", X, weight, SkipIn,
                        axis_i = axis, eps_f = eps, skip_term_i = True,
                        outputs = 2)
        return Y.setTypeAs(X), SkipOut.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor, SkipIn: torch.Tensor = None,
        axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return X, X
        if SkipIn is None:
            SkipOut = X
        else:
            SkipOut = X + SkipIn
        x = SkipOut.float()
        mean_square = x.pow(2).mean(axis, keepdim=True)
        Y = x * torch.rsqrt(mean_square + eps)
        Y = Y.type_as(X) * weight
        return Y, SkipOut


def rms_norm(X: torch.Tensor, weight: torch.Tensor,
        axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
    return RMSNorm.apply(X, weight, axis, eps)


def skip_rms_norm(X: torch.Tensor, weight: torch.Tensor, SkipIn: torch.Tensor = None,
        axis: int = -1, eps: float = 1e-5) -> torch.Tensor:
    return SkipRMSNorm.apply(X, weight, SkipIn, axis, eps)


if __name__ == "__main__":
    torch.manual_seed(1)
    
    class TestModule1(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))


        def forward(self, X: torch.Tensor):
            return rms_norm(X, self.weight, -1, self.eps)


    class TestModule2(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.randn(dim, dtype=torch.float16))


        def forward(self, X: torch.Tensor, SkipIn: torch.Tensor):
            return skip_rms_norm(X, self.weight, SkipIn, -1, self.eps)

    # test_op1 = TestModule1(4096, 1e-6)

    name ="case3"
    seq  = 8
    dim  = 2048
    
    folder = '../models/RMSNorm/{}'.format(name)
    if not os.path.exists(folder):
        Path().mkdir(parents=True)
    

    test_op2 = TestModule2(dim, 1e-6)

    input = torch.randn([seq, dim], dtype=torch.float16)
    skip = torch.randn([seq, dim], dtype=torch.float16)

    output1, output2 = test_op2.forward(input, skip)
    input.numpy().tofile("../models/RMSNorm/{}/input-{}-{}.bin".format(name, ModelUtils.getShape(input), ModelUtils.getType(input)))
    skip.numpy().tofile("../models/RMSNorm/{}/skip-{}-{}.bin".format(name, ModelUtils.getShape(skip), ModelUtils.getType(skip)))

    output1.detach().numpy().tofile("../models/RMSNorm/{}/output1.bin".format(name))
    output2.detach().numpy().tofile("../models/RMSNorm/{}/output2.bin".format(name))

    torch.onnx.export(test_op2, (input, skip), "../models/RMSNorm/{}/model.onnx".format(name),
       input_names=["input", "skip"], output_names=["output1","output2"], opset_version=11)
