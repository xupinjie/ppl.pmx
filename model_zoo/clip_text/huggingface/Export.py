import fire
import sys
import os
import json
import torch
torch.set_default_tensor_type(torch.FloatTensor)

from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import clip_text.modeling.Loader as Loader
from ModelParams import ClipTextModelParams

def main(
    ckpt_dir: str,
    export_path: str,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
):
    with open(Path(ckpt_dir) / "pmx_text_params.json", "r") as f:
        params = json.loads(f.read())
    params: ClipTextModelParams = ClipTextModelParams(**params)

    model = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, auto_causal,
        True, True, True, True
    )

    # export model
    input_ids = torch.ones([1, 77], dtype=torch.int32)

    # to do: dynamic batch / dump json
    torch.onnx.export(
        model.cpu(),
        (input_ids),
        os.path.join(export_path, "model.onnx"),
        input_names=["input_ids"],
        output_names=["text_embeddings"],
        do_constant_folding=True,
        opset_version=11,
    )

# torchrun Export.py -ckpt_dir ../models/ --export_path ../models
if __name__ == "__main__":
    fire.Fire(main)
