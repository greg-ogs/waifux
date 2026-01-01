"""
Container for Gemma3-1b or slower
"""

from transformers import AutoTokenizer, Gemma3ForCausalLM, TextStreamer
import torch
import os
import time

model_id = "google/gemma-3-1b-it"

hf_token = os.getenv("HF_TOKEN")

custom_cache_dir = "model_files/Gemma3"

if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

model = Gemma3ForCausalLM.from_pretrained(
    model_id, token=hf_token, cache_dir=custom_cache_dir, dtype=torch.bfloat16
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

streamer = TextStreamer(tokenizer, skip_prompt=True)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who was the last f1 champion?"},]
        },
    ],
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, streamer=streamer, temperature=0.3)

# outputs = tokenizer.batch_decode(outputs)
