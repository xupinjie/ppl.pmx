import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")  # 添加上一级目录到sys.path
from model_zoo import ModelUtils

class ColumnParallelLinear(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, W: torch.Value, B: torch.Value, proc_group: torch.Value,
        in_features: int, out_features: int, gather_output: bool = True):
        if B is not None:
            Y = g.op("pmx::ColumnParallelLinear", X, W, B,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = True,
                    gather_output_i = gather_output)
        else:
            Y = g.op("pmx::ColumnParallelLinear", X, W,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = False,
                    gather_output_i = gather_output)
        return Y


    @staticmethod
    def forward(
        self, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
        in_features: int, out_features: int, gather_output: bool = True):
        if torch.onnx.is_in_onnx_export():
            output_parallel = torch.zeros(*X.shape[:-1], W.shape[0], dtype=W.dtype).to(X.device)
            if gather_output and proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                last_dim = output_parallel.dim() - 1
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                tensor_list = [torch.zeros_like(output_parallel) for _ in range(world_size)]
                tensor_list[rank] = output_parallel
                Y = torch.cat(tensor_list, dim=last_dim).contiguous()
            else:
                Y = output_parallel
            return Y
        else:
            # Matrix multiply.
            output_parallel = F.linear(X, W, B)
            # All-gather across the partitions.
            if gather_output and proc_group is not None and torch.distributed.get_world_size(proc_group) > 1:
                last_dim = output_parallel.dim() - 1
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                tensor_list = [torch.empty_like(output_parallel) for _ in range(world_size)]
                tensor_list[rank] = output_parallel
                torch.distributed.all_gather(tensor_list, output_parallel, group=proc_group)
                Y = torch.cat(tensor_list, dim=last_dim).contiguous()
            else:
                Y = output_parallel
        return Y


def column_parallel_linear(
        X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, proc_group: dist.ProcessGroup,
        in_features: int, out_features: int, gather_output: bool = True) -> torch.Tensor:
    return ColumnParallelLinear.apply(X, W, B, proc_group, in_features, out_features, gather_output)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            proc_group: dist.ProcessGroup,
            in_features: int,
            out_features: int,
            bias_term: bool = True,
            gather_output: bool = True) -> None:
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.gather_output = gather_output
            self.proc_group = proc_group

            world_size = 1 if proc_group is None else proc_group.size()
            assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

            self.out_features_per_partition = out_features // world_size

            self.weight = nn.Parameter(torch.randn(self.out_features_per_partition, self.in_features, dtype=torch.float16))
            if bias_term:
                self.bias = nn.Parameter(torch.randn(self.out_features_per_partition, dtype=torch.float16))
            else:
                self.register_parameter("bias", None)


        def forward(self, X: torch.Tensor):
            return column_parallel_linear(
                X, self.weight, self.bias, self.proc_group,
                self.in_features, self.out_features)

    torch.manual_seed(1)
    name = "case2"
    seq_len = 16
    in_feature = 4096
    ou_feature = 4096

    test_op1 = TestModule1(None, in_feature, ou_feature, False, False)

    input = torch.randn([seq_len, in_feature], dtype=torch.float16)
    output = test_op1.forward(input)

    TensorDumper = ModelUtils.__TensorDumperV2__("../models/ColumnParallelLinear/{}".format(name))
    TensorDumper.dump(input, "input".format(name))
    TensorDumper.dump(output.detach(), "output".format(name))
    torch.onnx.export(test_op1, (input), "../models/ColumnParallelLinear/{}/model.onnx".format(name), opset_version=11)

    # model_str1 = torch.onnx.export_to_pretty_string(
    #     test_op1, (input), "ColumnParallelLinear1.onnx", opset_version=11)

    # print(model_str1)
