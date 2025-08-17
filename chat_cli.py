# chat_cli.py (Improved Version)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 1. Load the model and tokenizer
    model_path = "./nanollm-final-model"
    print("Loading model... This might take a moment.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}. Type 'exit' or 'quit' to end the chat.")
    print("-" * 30)

    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Initialize chat history as a string
    chat_history = ""

    # 3. Start the chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Assistant: Goodbye!")
            break

        # 4. Construct the prompt with history
        # We create a single block of text with a clear pattern
        prompt = f"{chat_history}You: {user_input}\nAssistant:"

        # 5. Tokenize the entire prompt
        # This method automatically creates the 'attention_mask' for us
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # 6. Generate a response
        outputs = model.generate(
            **inputs, # Pass both input_ids and attention_mask
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 7. Decode only the newly generated tokens
        new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        print(f"Assistant: {response}")

        # 8. Update the history for the next turn
        chat_history += f"You: {user_input}\nAssistant: {response}\n"

        # --- Context Window Management ---
        # (This part is harder with text, but for a simple demo this is okay.
        # For long chats, the prompt would need to be trimmed.)

if __name__ == "__main__":
    main()
