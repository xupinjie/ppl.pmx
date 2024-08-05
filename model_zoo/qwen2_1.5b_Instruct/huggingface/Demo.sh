OMP_NUM_THREADS=1 torchrun \
        --nproc_per_node 1 \
        Demo.py \
        --ckpt_dir ../models \
        --tokenizer_path ~/.cache/modelscope/hub/qwen/Qwen2-1___5B-Instruct/tokenizer.json \
        --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1