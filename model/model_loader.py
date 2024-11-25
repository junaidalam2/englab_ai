from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned-spiritual")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned-spiritual")

role_prompt = "You are a spiritual teacher who guides people on how to meditate, breathe consciously, and live mindfully."

def generate_response(prompt):
    print("User prompt:", prompt)
    # Encode the input text
    inputs = tokenizer(role_prompt + " " + prompt, return_tensors="pt")

    # Generate text
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,  # Adjust for longer responses
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
    )

    # Decode the generated text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"text": response_text, "status": "success"}
