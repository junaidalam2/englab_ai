import os
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load environment variables from .env file
load_dotenv()

role_prompt = "You are a helpful assistant who follows instructions explicitly and does not hallucinate."

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_response(prompt):
    # Encode the input text
    inputs = tokenizer(role_prompt + " " + prompt, return_tensors="pt")

    # Generate text
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,  # Adjust as needed
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
    )

    # Decode the generated text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"text": response_text, "status": "success"}

if __name__ == "__main__":
    # Example prompts to test the function
    prompts = [
        "What is the capital of France?",
        "Explain quantum physics in simple terms.",
        "What are the benefits of meditation?",
    ]

    for prompt in prompts:
        response = generate_response(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response['text']}\n")
