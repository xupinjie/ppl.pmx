import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as PMX
from ModelParams import ModelParams, VisionModelParams
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from ModelLayers import Linear

TensorDumper = ModelUtils.__TensorDumper__()

class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return PMX.layer_norm(x, self.weight, self.bias, -1, self.eps)

class SkipLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, X: torch.Tensor, SkipIn: torch.Tensor):
        return PMX.skip_layer_norm(X, self.weight, self.bias, SkipIn, -1, self.eps)

class Attention(nn.Module):
    def __init__(
            self,
            args: ModelParams,
            layer_id: int,
            friendly_gqa: bool,
            fused_qkv: bool,
            attn_wqkv_bias_term: bool,
            attn_wo_bias_term: bool,
            proc_group: dist.ProcessGroup):
        super().__init__()

        world_size = 1 if proc_group is None else proc_group.size()

        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        self.num_local_heads = args.num_heads // world_size
        self.num_local_kv_heads = self.num_kv_heads // world_size
        self.num_local_kv_repeats = self.num_local_heads // self.num_local_kv_heads
        self.head_dim = args.hidden_dim // args.num_heads
        self.num_layers = args.num_layers
        self.layer_id = layer_id
        self.friendly_gqa = friendly_gqa
        self.fused_qkv = fused_qkv
        self.auto_causal = args.auto_causal

        if self.fused_qkv:
            self.wqkv = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim + 2 * self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
        else:
            self.wq = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
            self.wk = ColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
            self.wv = ColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
        self.wo = RowParallelLinear(
            proc_group, args.hidden_dim, args.hidden_dim,
            bias_term=attn_wo_bias_term, input_is_parallel=True)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        expanded_shape = (0, 0, -1, self.head_dim)
        if self.fused_qkv:
            xqkv = self.wqkv(x)
            xqkv = PMX.reshape(xqkv, expanded_shape)
            # TensorDumper.dump(xqkv, "layer{}_ColumnParallelLinear0_".format(self.layer_id))
            split_size = (self.num_local_heads, self.num_local_kv_heads, self.num_local_kv_heads)
            xq, xk, xv = torch.split(xqkv, split_size, -2)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            xq = PMX.reshape(xq, expanded_shape)
            xk = PMX.reshape(xk, expanded_shape)
            xv = PMX.reshape(xv, expanded_shape)
        # TensorDumper.dump(xq, "layer{}_reshaped_xq".format(self.layer_id))
        # TensorDumper.dump(xk, "layer{}_reshaped_xk".format(self.layer_id))
        # TensorDumper.dump(xv, "layer{}_reshaped_xv".format(self.layer_id))


        attn = PMX.multi_head_attention(xq, xk, xv,
                                        attn_mask=attn_mask,
                                        num_heads=self.num_local_heads,
                                        head_dim=self.head_dim,
                                        is_causal=self.auto_causal,
                                        num_kv_heads=0 if self.friendly_gqa else self.num_local_kv_heads)
        # TensorDumper.dump(attn, "layer{}_multi_head_attention_out".format(self.layer_id))

        output = self.wo(PMX.reshape(attn, (0, 0, -1)))
        # TensorDumper.dump(output, "layer{}_reshaped_wo_out".format(self.layer_id))

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelParams,
        layer_id: int,
        linear_bias_term: bool,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()
        self.layer_id = layer_id

        self.w1 = ColumnParallelLinear(
            proc_group, args.hidden_dim, args.intermediate_dim,
            bias_term=linear_bias_term, gather_output=False)
        self.w2 = RowParallelLinear(
            proc_group, args.intermediate_dim, args.hidden_dim,
            bias_term=linear_bias_term, input_is_parallel=True)


    def forward(self, x):
        x1 = self.w1(x)
        # x1 = PMX.quickgelu(x1)
        x1 = PMX.swish(x1, beta=1.702)
        # TensorDumper.dump(x1, "layer{}_ffn_w1".format(self.layer_id))
        output = self.w2(x1)
        # TensorDumper.dump(output, "layer{}_ffn_w2".format(self.layer_id))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 args: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.attention = Attention(args,
                                   layer_id,
                                   friendly_gqa,
                                   fused_qkv,
                                   attn_wqkv_bias_term,
                                   attn_wo_bias_term,
                                   proc_group=proc_group)
        self.feed_forward = FeedForward(args,
                                        layer_id,
                                        ffn_linear_bias_term,
                                        proc_group=proc_group)

        self.layer_id = layer_id
        self.attention_norm = SkipLayerNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = SkipLayerNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, skip: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        norm, res1 = self.attention_norm(x, skip) # res1 = input_x when skip==None
        # TensorDumper.dump(norm, "layer{}_norm0_out0".format(self.layer_id))
        # TensorDumper.dump(res1, "layer{}_norm0_out1".format(self.layer_id))
        attn = self.attention.forward(norm, attn_mask)
        norm, res2 = self.ffn_norm(attn, res1) # res2 = attn + res1
        # TensorDumper.dump(norm, "layer{}_norm1_out0".format(self.layer_id))
        # TensorDumper.dump(res2, "layer{}_norm1_out1".format(self.layer_id))
        ffn = self.feed_forward.forward(norm)
        return ffn, res2

class CLIPTextEmbeddings(nn.Module):
    def __init__(self, hidden_size:int, vocab_size:int, max_position_embeddings:int, proc_group: dist.ProcessGroup):
        super().__init__()
        embed_dim = hidden_size

        # self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim)
        self.token_embedding = ParallelEmbedding(proc_group=proc_group, num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = ParallelEmbedding(proc_group=proc_group, num_embeddings=max_position_embeddings, embedding_dim=embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, #[2, 7]
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2] 

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length] #[1, 7]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids) #[2, 7, 768]

        position_embeddings = self.position_embedding(position_ids) #[1, 7, 768]
        embeddings = inputs_embeds + position_embeddings #[2, 7, 768]

        # TensorDumper.dump(inputs_embeds, "inputs_embeds")
        # TensorDumper.dump(position_embeddings, "position_embeddings")

        return embeddings

class ClipTextTransformer(nn.Module):
    def __init__(self, params: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.params = params
        self.n_layers = params.num_layers
        self.proc_group = proc_group
        self.fused_qkv = fused_qkv

        embed_dim = params.hidden_dim

        world_size = 1 if proc_group is None else proc_group.size()
        num_kv_heads = params.num_heads if params.num_kv_heads is None else params.num_kv_heads
        num_local_heads = params.num_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = params.hidden_dim // params.num_heads
        self.local_q_dim = num_local_heads * head_dim
        self.local_kv_dim = num_local_kv_heads * head_dim

        self.embeddings = CLIPTextEmbeddings(params.hidden_dim, params.vocab_size, params.max_position_embeddings, proc_group=proc_group)
        self.final_layer_norm = LayerNorm(embed_dim, eps=params.norm_eps)
        # "hidden_size": 512, "vocab_size": 49408, max_position_embeddings": 77

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(
                layer_id, params,
                friendly_gqa,
                fused_qkv,
                attn_wqkv_bias_term,
                attn_wo_bias_term,
                ffn_linear_bias_term,
                proc_group=proc_group))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor):
        # input_ids: shape[2, 7]
        h = self.embeddings(input_ids=input_ids, position_ids=None)
        TensorDumper.dump(h, "pmx_embeddings")

        norm = None
        for layer in self.layers:
            h, norm = layer(h, norm, attn_mask)

        last_hidden_state = norm+h
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        TensorDumper.dump(last_hidden_state, "pmx_last_hidden_state_output")

        return last_hidden_state

        #TODO: TO add this layer
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        # pooled_output = last_hidden_state[
        #     torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        #     input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        # ]

        # TensorDumper.dump(pooled_output, "pmx_output")
        # return pooled_output

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        loaded_params = set()
        model_params = {key: value for key, value in self.named_parameters()}

        for key, value in state_dict.items():
            module_name, param_name = key.rsplit(".", 1)

            if key in model_params:
                self.get_submodule(module_name)._parameters[param_name][:] = value
                loaded_params.add(key)
                print(f'Loaded: {key} -> {key}[{value.shape}]')

            try:
                if self.fused_qkv:
                    if 'attention.wq' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wq', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            :self.local_q_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wk' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wk', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_q_dim:self.local_q_dim + self.local_kv_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wv' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wv', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_q_dim + self.local_kv_dim:
                            self.local_q_dim + self.local_kv_dim * 2] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')

        for key in state_dict:
            if key not in loaded_params:
                print(f'{key} is not loaded.')
