import torch

from typing import Optional

if __name__ == "__main__":
    from KeyValueCache import key_value_cache
    from MultiHeadAttention import multi_head_attention
else:
    from .KeyValueCache import key_value_cache
    from .MultiHeadAttention import multi_head_attention
import sys
sys.path.append("../..")  # 添加上一级目录到sys.path
from model_zoo import ModelUtils
TensorDumper = ModelUtils.__TensorDumperV2__()

class MultiHeadCacheAttention(torch.autograd.Function):
    @staticmethod
    def symbolic(g, query: torch.Value, current_key: torch.Value, current_value: torch.Value,
                 seqstarts: torch.Value, kvstarts: torch.Value, cachestarts: torch.Value,
                 start_pos: torch.Value, decoding_batches: torch.Value,
                 max_seqlen: torch.Value, max_kvlen: torch.Value,
                 cache: torch.Value, scale: Optional[torch.Value],
                 attn_mask: Optional[torch.Value], num_heads: int, head_dim: int,
                 is_causal: bool = True, num_kv_heads: int = 0,
                 num_layer: int = 1, layer_idx: int = 0,
                 quant_bit: int = 0, quant_group: int = 8,
                 cache_mode: int = 0, cache_layout: int = 0):
        # g: GraphContext, defined in onnx/_internal/jit_utils.py
        if attn_mask is not None:
            output = g.op('pmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache, scale, attn_mask,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                num_kv_heads_i=num_kv_heads,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout)
        elif scale is not None:
            output = g.op('pmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen,
                cache, scale,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                num_kv_heads_i=num_kv_heads,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout)
        else:
            output = g.op('pmx.dynamic_batching::MultiHeadCacheAttention',
                query, current_key, current_value,
                seqstarts, kvstarts, cachestarts,
                start_pos, decoding_batches,
                max_seqlen, max_kvlen, cache,
                num_heads_i=num_heads,
                head_dim_i=head_dim,
                is_causal_i=is_causal,
                num_kv_heads_i=num_kv_heads,
                num_layer_i=num_layer,
                layer_idx_i=layer_idx,
                quant_bit_i=quant_bit,
                quant_group_i=quant_group,
                cache_mode_i=cache_mode,
                cache_layout_i=cache_layout)
        return output.setTypeAs(query)


    @staticmethod
    def forward(ctx, query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                 seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                 start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                 max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                 cache: torch.Tensor, scale: Optional[torch.Tensor],
                 attn_mask: Optional[torch.Tensor], num_heads: int, head_dim: int,
                 is_causal: bool = True, num_kv_heads: int = 0,
                 num_layer: int = 1, layer_idx: int = 0,
                 quant_bit: int = 0, quant_group: int = 8,
                 cache_mode: int = 0, cache_layout: int = 0):
        if torch.onnx.is_in_onnx_export():
            return query

        key, value = key_value_cache(
            current_key, current_value,
            seqstarts, kvstarts, cachestarts,
            start_pos, max_seqlen, max_kvlen,
            cache, scale, num_layer, layer_idx,
            quant_bit, quant_group, 1,
            cache_mode, cache_layout)

        TensorDumper.dump(key.detach(), "key_value_cache_key")
        TensorDumper.dump(value.detach(), "key_value_cache_value")
        TensorDumper.dump(cache.detach(), "key_value_cache_cache")
        TensorDumper.dump(scale.detach(), "key_value_cache_scale")
        
        output = multi_head_attention(
            query, key, value, seqstarts,
            kvstarts, decoding_batches,
            max_seqlen, max_kvlen, attn_mask,
            num_heads, head_dim,
            is_causal, num_kv_heads)

        return output


def multi_head_cache_attention(
                query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                cache: torch.Tensor, scale: Optional[torch.Tensor],
                attn_mask: Optional[torch.Tensor], num_heads: int, head_dim: int,
                is_causal: bool = True, num_kv_heads: int = 0,
                num_layer: int = 1, layer_idx: int = 0,
                quant_bit: int = 0, quant_group: int = 8,
                cache_mode: int = 0, cache_layout: int = 0) -> torch.Tensor:
    if attn_mask is not None and scale is None:
        _scale = torch.empty(0, device=query.device)
    else:
        _scale = scale
    return MultiHeadCacheAttention.apply(query, current_key, current_value, seqstarts, kvstarts,
                                         cachestarts, start_pos, decoding_batches,
                                        max_seqlen, max_kvlen, cache, _scale,
                                        attn_mask, num_heads, head_dim,
                                        is_causal, num_kv_heads, num_layer,
                                        layer_idx, quant_bit, quant_group,
                                        cache_mode, cache_layout)


if __name__ == "__main__":
    # params: head, causual,  
    # attnmask, group not supported yet, layout todo
    num_heads = int (sys.argv[1]) if len(sys.argv) >= 2 else 32
    num_kv_heads = int (sys.argv[2]) if len(sys.argv) >= 3 else 32
    is_causal = int (sys.argv[3]) if len(sys.argv) >= 4 else 1
    seqlen = int (sys.argv[4]) if len(sys.argv) >= 5 else 1
    attn_mask_len = int (sys.argv[5]) if len(sys.argv) >= 6 else 0	
    prefill_flag = 0
    if seqlen>1:
        prefill_flag = 1
 
    #todo, fixed yet
    bs =1 #todo for bs>1 decoder
    head_dim = 128
    kvlen = 32
    num_layer = 2
    layer_idx = 1
    quant_group = 8
    quant_bit = 8
   
    class TestModule1(torch.nn.Module):
        def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, is_causal: bool = True,
                     num_layer: int = 1, layer_idx: int = 0,
                     quant_bit: int = 0, quant_group: int = 8) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.is_causal = is_causal
            print(" op cau is ",self.is_causal)
            self.num_layer = num_layer
            self.layer_idx = layer_idx
            self.quant_bit = quant_bit
            self.quant_group = quant_group


        def forward(self, query: torch.Tensor, current_key: torch.Tensor, current_value: torch.Tensor,
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                start_pos: torch.Tensor, decoding_batches: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                cache: torch.Tensor, scale: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
            return multi_head_cache_attention(
                                        query, current_key, current_value, seqstarts, kvstarts,
                                        cachestarts, start_pos, decoding_batches,
                                        max_seqlen, max_kvlen, cache, scale, attn_mask,
                                        self.num_heads, self.head_dim, self.is_causal, self.num_kv_heads,
                                        self.num_layer, self.layer_idx, self.quant_bit, self.quant_group,cache_layout=3)

    torch.manual_seed(1)
    name = "case2"


    q = torch.randn(bs * seqlen, num_heads, head_dim, dtype=torch.float16)
    k = torch.randn(bs * seqlen, num_kv_heads, head_dim, dtype=torch.float16)
    v = torch.randn(bs * seqlen, num_kv_heads, head_dim, dtype=torch.float16)
	
	#prefill,todo
    attn_mask = torch.randn(bs * seqlen, bs * seqlen, dtype=torch.float16)

    seqstarts = torch.tensor([0, seqlen], dtype=torch.int64).cumsum(dim=0)
    print(seqstarts)
    if prefill_flag>0:
        decoding_batches = torch.tensor([0], dtype=torch.int64)
    else :
        decoding_batches = torch.tensor([bs], dtype=torch.int64)

    cache = torch.zeros([ num_layer, 2, num_kv_heads, bs * kvlen,head_dim], dtype=torch.int8)
    scale = torch.zeros([ num_layer, 2, num_kv_heads, bs * kvlen,head_dim // quant_group], dtype=torch.float16)
    start_pos = torch.full([bs], 0, dtype=torch.int64)
    if prefill_flag==0:
        start_pos[0] = 8
    #print(start_pos)
    cachestarts = torch.arange(0, bs * kvlen, kvlen, dtype=torch.int64)
    print(cachestarts)

    kvstarts = torch.zeros([bs + 1], dtype=torch.int64)
    kvstarts[1:] = start_pos.cumsum(0)
    kvstarts = kvstarts + seqstarts
    print(kvstarts)

    max_seqlen = torch.tensor([seqlen])
    max_kvlen = torch.tensor([kvstarts[-1]])

    test_op1 = TestModule1(num_heads, num_kv_heads, head_dim, (is_causal>0), num_layer, layer_idx, quant_bit, quant_group)
    test_op2 = TestModule1(num_heads, num_kv_heads, head_dim, (is_causal>0), num_layer, layer_idx, 0, quant_group)

    if attn_mask_len>0:
    	output = test_op1.forward(q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale, attn_mask)
    else:
	    output = test_op1.forward(q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale)#, attn_mask)
    
        
    q.numpy().tofile(                              "../../models/MHACache/{}/0input_q-{}-{}.bin".format(name, ModelUtils.getShape(q), ModelUtils.getType(q)))
    k.numpy().tofile(                              "../../models/MHACache/{}/1input_k-{}-{}.bin".format(name, ModelUtils.getShape(k), ModelUtils.getType(k)))
    v.numpy().tofile(                              "../../models/MHACache/{}/2input_v-{}-{}.bin".format(name, ModelUtils.getShape(v), ModelUtils.getType(v)))
    seqstarts.numpy().tofile(              "../../models/MHACache/{}/3input_seqstarts-{}-{}.bin".format(name, ModelUtils.getShape(seqstarts), ModelUtils.getType(seqstarts)))
    kvstarts.numpy().tofile(                "../../models/MHACache/{}/4input_kvstarts-{}-{}.bin".format(name, ModelUtils.getShape(kvstarts), ModelUtils.getType(kvstarts)))
    cachestarts.numpy().tofile(          "../../models/MHACache/{}/5input_cachestarts-{}-{}.bin".format(name, ModelUtils.getShape(cachestarts), ModelUtils.getType(cachestarts)))
    start_pos.numpy().tofile(              "../../models/MHACache/{}/6input_start_pos-{}-{}.bin".format(name, ModelUtils.getShape(start_pos), ModelUtils.getType(start_pos)))
    decoding_batches.numpy().tofile("../../models/MHACache/{}/7input_decoding_batches-{}-{}.bin".format(name, ModelUtils.getShape(decoding_batches), ModelUtils.getType(decoding_batches)))
    max_seqlen.numpy().tofile(            "../../models/MHACache/{}/8input_max_seqlen-{}-{}.bin".format(name, ModelUtils.getShape(max_seqlen), ModelUtils.getType(max_seqlen)))
    max_kvlen.numpy().tofile(              "../../models/MHACache/{}/9input_max_kvlen-{}-{}.bin".format(name, ModelUtils.getShape(max_kvlen), ModelUtils.getType(max_kvlen)))
    cache.numpy().tofile(                     "../../models/MHACache/{}/10input_cache-{}-{}.bin".format(name, ModelUtils.getShape(cache), ModelUtils.getType(cache)))
    scale.numpy().tofile(                     "../../models/MHACache/{}/11input_scale-{}-{}.bin".format(name, ModelUtils.getShape(scale), ModelUtils.getType(scale)))
    
    if attn_mask_len>0:
	    attn_mask.numpy().tofile(             "../../models/MHACache/{}/12input_attn_mask-{}-{}.bin".format(name, ModelUtils.getShape(attn_mask), ModelUtils.getType(attn_mask)))
	
    
    output.detach().numpy().tofile("../../models/MHACache/{}/output.bin".format(name))

    if attn_mask_len>0:
    	model_str1 = torch.onnx.export(
       	    test_op1, (q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale,attn_mask),
            "../../models/MHACache/{}/model.onnx".format(name), opset_version=11)
    else:
	    model_str1 = torch.onnx.export(
       	    test_op1, (q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale),#,attn_mask),
            "../../models/MHACache/{}/model.onnx".format(name), opset_version=11)


    # model_str1 = torch.onnx.export_to_pretty_string(
    #    test_op1, (q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale),
    #    "MultiHeadAttention1.onnx", opset_version=11)
    # model_str2 = torch.onnx.export_to_pretty_string(
    #    test_op1, (q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, scale, attn_mask),
    #    "MultiHeadAttention2.onnx", opset_version=11)
    
    # cache = cache.to(q)
    # model_str3 = torch.onnx.export_to_pretty_string(
    #    test_op2, (q, k, v, seqstarts, kvstarts, cachestarts, start_pos, decoding_batches, max_seqlen, max_kvlen, cache, None, attn_mask),
    #    "MultiHeadAttention3.onnx", opset_version=11)

    # print(model_str1)
    # print(model_str2)
    # print(model_str3)
