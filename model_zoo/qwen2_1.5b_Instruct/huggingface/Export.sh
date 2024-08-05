OMP_NUM_THREADS=1 torchrun --nproc_per_node 1 \
Export.py --ckpt_dir ../models/  \
          --fused_qkv 1 --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 \
          --export_path ../models
