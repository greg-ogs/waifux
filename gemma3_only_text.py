"""
Container for Gemma3-1b or slower
"""
import fitz
from transformers import AutoTokenizer, Gemma3ForCausalLM, TextStreamer
import torch
import os
import time

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
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    model_id = "google/gemma-3-270m-it"

    hf_token = os.getenv("HF_TOKEN")

    custom_cache_dir = "model_files/Gemma3"

    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    dtype = torch.float32

    print(f"Using device: {device}")

    model = Gemma3ForCausalLM.from_pretrained(
        model_id, token=hf_token, cache_dir=custom_cache_dir, torch_dtype=dtype,  device_map=device
    ).eval()

    model.config._attn_implementation = "eager"
    # TORCH_DISABLE_DYNAMO=1

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    # Using small context
    # For larger files use Vector Database" (like ChromaDB or FAISS
    context_file = "context.txt"

    try:
        retrieved_info = retrieve_from_file(context_file)
    except FileNotFoundError:
        retrieved_info = "No additional context found."

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are a helpful and cute waifu. Use the following information to answer the user:\n\n{retrieved_info}"}, ]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Hello, how are you? Can you tell me something about the F1?"
                                                     f""}, ]
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

    time_0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200, min_new_tokens=30, repetition_penalty=1.3, streamer=streamer,
                                 use_cache=False,
                                 eos_token_id=terminators,
                                 do_sample=True, temperature=0.6, top_p=0.3,
                                 )

    # outputs = tokenizer.batch_decode(outputs)

    time_1 = time.time()
    print(f"Total inference time is:  {time_1 - time_0:.4f} seconds")
