import fire
import sys
import os
import json
import torch
import numpy as np

from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import clip_text.modeling.Loader as Loader
from ModelParams import ClipTextModelParams

def ref():
    from transformers import CLIPTokenizer, CLIPTextModel

    # model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    x =last_hidden_state.detach().numpy().tofile("output.bin")

    return last_hidden_state, pooled_output

def main(
    ckpt_dir: str,
    batch: int = 1,
    friendly_gqa: bool = False, # done gqa by repeating key and value by key_value_cache op
    fused_qkv: bool = True, # fuse qkv linear
    auto_causal: bool = True, # causal mask is auto done by attention op, no need to pass additional mask to the model
    dump_tensor_path: str = None,
    dump_steps: List[int] = []
):
    torch.set_default_tensor_type(torch.HalfTensor)

    with open(Path(ckpt_dir) / "pmx_text_params.json", "r") as f:
        params = json.loads(f.read())
    params: ClipTextModelParams = ClipTextModelParams(**params)

    model = Loader.load(
        ckpt_dir, params, friendly_gqa,
        fused_qkv, auto_causal,
        True, True, True, False,
        dump_tensor_path, dump_steps
    )


    torch.set_default_tensor_type(torch.HalfTensor)

    # input_ids = torch.randint(0, 10, (2, 7), dtype=torch.int64)
    input_ids = np.fromfile("./datas/step0_input_ids-2_7-int64.bin", dtype=np.int64)
    input_ids = input_ids.reshape((2, 7))
    input_ids = torch.from_numpy(input_ids)
    # print(input_ids)

    # attn_mask = torch.empty(0, dtype=torch.float32)
    attn_mask = np.fromfile("./datas/step0_attention_mask-2_1_7_7-fp32.bin", dtype=np.float32)
    attn_mask = attn_mask.reshape((2, 1, 7, 7)) + np.zeros((2, 8, 7, 7))
    attn_mask = torch.from_numpy(attn_mask)
    # print(attn_mask)

    outputs = model.forward(input_ids, attn_mask)

if __name__ == "__main__":
    # ref()
    fire.Fire(main)

# torchrun Demo.py ../models
