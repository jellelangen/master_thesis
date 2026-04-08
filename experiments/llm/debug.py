from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.float16,attn_implementation="eager")

messages = [{"role": "user", "content": "What colour is lapis lazuli?\nAnswer in a few words."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to("cuda:0")
out = model.generate(**inputs, max_new_tokens=50, output_attentions=True, return_dict_in_generate=True)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))