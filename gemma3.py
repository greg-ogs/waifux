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

# from TTS.api import TTS
# import sounddevice as sd
# import soundfile as sf

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

    torch.set_float32_matmul_precision('high')

    model_id = "google/gemma-3-4b-it"

    hf_token = os.getenv("HF_TOKEN")

    custom_cache_dir = "model_files/Gemma3"

    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable not found. Please set it before running.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Using device: {device}")

    # tts = TTS("tts_models/en/jenny/jenny").to(device)

    model_kwargs = {}

    if "nv" in torch.__version__:
        print("Detected Jetson/Nvidia specific PyTorch version. Using 'torch_dtype'.")
        model_kwargs["torch_dtype"] = dtype
    else:
        print("Detected standard PyTorch version. Using 'dtype'.")
        # Note: Standard Transformers usually expects 'torch_dtype', but satisfying user request for 'dtype'
        model_kwargs["dtype"] = dtype


    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, token=hf_token, cache_dir=custom_cache_dir, torch_dtype=dtype,  device_map=device
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token, cache_dir=custom_cache_dir, use_fast=True)

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Using small context
    # For larger files use Vector Database" (like ChromaDB or FAISS
    context_file = "context.txt"

    try:
        retrieved_info = retrieve_from_file(context_file)
    except FileNotFoundError:
        retrieved_info = "No additional context found."

    messages = [{
        "role": "system",
        "content": [{"type": "text",
                     "text": f"You are a helpful and cute waifu."}, ]
    }, {"role": "user", "content": [{"type": "text", "text": f"Please, tell me in the cutest way tht the system is "
                                                             f"loading, something like 'stand by' or 'wait a minute please' "}]}]

    # Add user input as a starting point

    # Use the following information to answer the user:\n\n{retrieved_info}

    print("\n--- Gemma 3 Chat initialized (Type 'exit' to quit) ---")
    count=0
    while True:
        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            time_0 = time.time()

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=500, min_new_tokens=30, do_sample=True, streamer=streamer, temperature=0.3)
                new_tokens = generation[0][input_len:]
                decoded_response = processor.decode(new_tokens, skip_special_tokens=True)
                decoded_response = decoded_response.replace("<end_of_turn>", "").strip()

            time_1 = time.time()
            # print(f"Total inference time is:  {time_1 - time_0:.4f} seconds")
            messages.append({"role": "model", "content": [{"type": "text", "text": decoded_response}]})

            if count==0:
                messages.append({"role": "user", "content": [{"type": "text", "text": f"Say hello in the cutest way"}]})
                count = 1
            else:
                user_input = input("\nUser: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

            # if decoded_response:
            #     audio_file = "response.wav"
            #
            #     tts.tts_to_file(text=decoded_response, file_path=audio_file)
            #
            #     data, samplerate = sf.read(audio_file)
            #     sd.play(data, samplerate)
            #     sd.wait()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            raise e
