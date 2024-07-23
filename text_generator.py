from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
import time

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for Apple silicon, use "mps" instead of "cuda"
print(f"Device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

start_time = time.time()

# Define the model name
# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = "SweatyCrayfish/llama-3-8b-quantized"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# Add a pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)

print(f"Initialization took {time.time() - start_time} seconds")

while True:
    text = input("User input: \n")
    if text.lower() == "exit":
        break
    start_time = time.time()

    # Encode the input text with attention mask
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate tokens with pad_token_id set
    max_length = 50  # Set a maximum length for generation
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.pad_token_id)
    
    print("Model output: \n")
    # Print tokens as they are generated
    for token_id in output_ids[0]:
        print(tokenizer.decode([token_id], skip_special_tokens=True), end="", flush=True)
    print()

    print(f"Generation took {time.time() - start_time} seconds")
