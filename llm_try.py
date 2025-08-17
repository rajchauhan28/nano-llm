# llm_try.py (revised)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define the path and load the model and tokenizer directly
model_path = "./nanollm-final-model"
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
print("Model and tokenizer loaded successfully!")

# Move the model to the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# 2. Define the prompt and tokenize it
prompt = "sampling.."
# The tokenizer converts the text prompt into numerical token IDs
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 3. Generate text using model.generate()
# This gives us direct control and avoids pipeline defaults.
print("Generating text...")
output_sequences = model.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=200,  # Explicitly generate 50 NEW tokens
    do_sample=True,     # Use sampling for more creative outputs
    top_k=50,
    top_p=0.95
)

# 4. Decode the output and print it
# The tokenizer converts the generated token IDs back into text
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("\n--- Generated Text ---")
print(generated_text)
