from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2-1.5B-Instruct",
    # "/home/SENSETIME/xupinjie1/.cache/modelscope/hub/qwen/Qwen2-1___5B-Instruct", torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-1.5B-Instruct")

print(model.state_dict().keys())

torch.save(model.state_dict(), "mm.pth")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
# Generate
generate_ids = model.generate(inputs.input_ids.cuda(), max_length=30)
res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)
# "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."

#/home/SENSETIME/xupinjie1/.cache/modelscope/hub/qwen/Qwen2-1___5B-Instruct


# {'architectures': ['Qwen2ForCausalLM'], 'attention_dropout': 0.0, 'bos_token_id': 151643, 'eos_token_id': 151645, 'hidden_act': 'silu', 'hidden_size': 1536, 'initializer_range': 0.02, 'intermediate_size': 8960, 'max_position_embeddings': 32768, 'max_window_layers': 21, 'model_type': 'qwen2', 'num_attention_heads': 12, 'num_hidden_layers': 28, 'num_key_value_heads': 2, 'rms_norm_eps': 1e-06, 'rope_theta': 1000000.0, 'sliding_window': 32768, 'tie_word_embeddings': True, 'torch_dtype': 'bfloat16', 'transformers_version': '4.40.1', 'use_cache': True, 'use_sliding_window': False, 'vocab_size': 151936}