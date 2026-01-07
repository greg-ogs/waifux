"""
Container for Gemma3
"""
import time
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextStreamer
from PIL import Image
import requests
import torch
import os
import fitz

def retrieve_from_file(file_path):
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    else:  # Assume text file
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


if __name__ == "__main__":
    model_id = "google/gemma-3-4b-it"

    hf_token = os.getenv("HF_TOKEN")

    custom_cache_dir = "model_files/Gemma3"

    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, token=hf_token, cache_dir=custom_cache_dir, dtype=dtype,  device_map=device
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token, cache_dir=custom_cache_dir, use_fast=True)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)

    # Using small context
    # For larger files use Vector Database" (like ChromaDB or FAISS
    context_file = "context.txt"

    try:
        retrieved_info = retrieve_from_file(context_file)
    except FileNotFoundError:
        retrieved_info = "No additional context found."

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/home/greg/PycharmProjects/waifux/drivers.jpg"},
                {"type": "text", "text": f"Who was the last f1 champion? and why is considered the best F1 "
                            f"driver in the world? Use the following information to answer the user:\n\n{retrieved_info}"}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=dtype)

    input_len = inputs["input_ids"].shape[-1]

    time_0 = time.time()

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500, do_sample=True, streamer=streamer, temperature=0.3)
        generation = generation[0][input_len:]

    # decoded = processor.decode(generation, skip_special_tokens=True)
    # print(decoded)
    time_1 = time.time()
    print(f"Total inference time is:  {time_1 - time_0:.4f} seconds")
