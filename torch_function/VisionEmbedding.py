import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")  # 添加上一级目录到sys.path
from model_zoo import ModelUtils

class VisionEmbedding(torch.autograd.Function):
    @staticmethod
    def symbolic(g, pixel_values: torch.Value, class_weight: torch.Value,
                 patch_weight: torch.Value, position_weight: torch.Value,
                 hidden_dim: int, patch_size: int):
        output = g.op('opmx::VisionEmbedding',
                      pixel_values, class_weight,
                      patch_weight, position_weight,
                      hidden_dim_i=hidden_dim,
                      patch_size_i=patch_size)
        return output


    @staticmethod
    def forward(self, pixel_values: torch.Value, class_weight: torch.Value,
                patch_weight: torch.Value, position_weight: torch.Value,
                hidden_dim: int, patch_size: int):
        num_patches = (pixel_values.shape[-1] // patch_size) * (pixel_values.shape[-2] // patch_size)
        print("num_patches:", num_patches)

        if torch.onnx.is_in_onnx_export():
            output = torch.zeros([pixel_values.shape[0], num_patches + 1, hidden_dim]).to(pixel_values.device)
            return output
        else:
            print(pixel_values.shape)
            num_positions = num_patches + 1
            position_ids = torch.arange(num_positions).expand((1, -1)).to(position_weight.device)
            print("position_ids:", position_ids.shape)
            print("position_ids:", position_ids)
            batch_size = pixel_values.shape[0]

            # embedding patch
            patch_embeds = F.conv2d(pixel_values, patch_weight, stride=patch_size) # shape -> [batch_size, hidden_dim, grid, grid]
            print("patch_embeds", patch_embeds.shape)
            TensorDumper = ModelUtils.__TensorDumperV2__("opmx_ops/visionembedding/ut1")
            TensorDumper.dump(patch_embeds, "convout")
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # shape -> [batch_size, grid*grid, hidden_dim]
            print("patch_embeds", patch_embeds.shape)
            
            # embedding class
            print("class_weight:", class_weight.shape)
            cls_embeds  = class_weight.expand(batch_size, 1, -1)
            print("cls_embeds:", cls_embeds.shape)
            
            # embedding position
            print("position_weight", position_weight.shape)
            print("position_weight", position_weight)
            pos_embeds = F.embedding(position_ids, position_weight)
            print("pos_embeds", pos_embeds.shape)
            print("pos_embeds:", pos_embeds)

            print("check:", (pos_embeds - position_weight == 0).all().item(), pos_embeds - position_weight)

            # merge embddding
            embeddings = torch.cat([cls_embeds, patch_embeds], dim=1) + pos_embeds # shape -> [batch_size, grid*grid + 1, hidden_dim]
            print(embeddings.shape)
            return embeddings


def vision_embedding(pixel_values: torch.Value, class_weight: torch.Value,
                     patch_weight: torch.Value, position_weight: torch.Value,
                     hidden_dim: int, patch_size: int) -> torch.Tensor:
    return VisionEmbedding.apply(pixel_values, class_weight, patch_weight, position_weight,
                                 hidden_dim, patch_size)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            image_size: int,
            patch_size: int):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_positions = (self.image_size // self.patch_size) ** 2 + 1

            self.class_weight = nn.Parameter(torch.randn(self.hidden_dim, dtype=torch.float16))
            self.patch_weight  = nn.Parameter(torch.randn([self.hidden_dim, 3, patch_size, patch_size], dtype=torch.float16))
            self.position_weight = nn.Parameter(torch.randn(self.num_positions, self.hidden_dim, dtype=torch.float16))

        def forward(self, pixel_values: torch.Tensor):
            return vision_embedding(pixel_values, self.class_weight, self.patch_weight, self.position_weight, self.hidden_dim, self.patch_size)

    torch.manual_seed(1)

    out_dir = "opmx_ops/visionembedding/ut1"
    batch, ic = 1, 3
    hidden_dim, image_size, patch_size = 512, 224, 32
    test_op1 = TestModule1(hidden_dim, image_size, patch_size)
    # pixel_values = torch.ones([1, 3, 224, 224])
    pixel_values = torch.randn([batch, ic, image_size, image_size], dtype=torch.float16)
    output = test_op1.forward(pixel_values)
    
    TensorDumper = ModelUtils.__TensorDumperV2__(out_dir)
    TensorDumper.dump(pixel_values, "input")
    TensorDumper.dump(output.detach(), "ref")

    # torch.onnx.export(
    #     test_op1,
    #     (pixel_values),
    #     "opmx_ops/visionembedding/ut1/model.onnx",
    #     input_names=['pixel_values'],
    #     output_names=['vision_embeddings'],
    #     opset_version=11,
    # )

    # model_str1 = torch.onnx.export_to_pretty_string(
    #     test_op1,  (pixel_values), "vision_embedding.onnx", opset_version=11)
    # print (model_str1)
    #out = test_op1.forward(pixel_values)

    # torch.onnx.export(
    #     test_op1,
    #     (pixel_values),
    #     "vision_embedding.onnx",
    #     input_names=['pixel_values'],
    #     output_names=['vision_embeddings'],
    #     opset_version=11,
    # )
