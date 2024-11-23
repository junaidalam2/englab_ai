from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def generate_response(prompt):
    # Encode the input text
    inputs = tokenizer(prompt, return_tensors="pt")

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
