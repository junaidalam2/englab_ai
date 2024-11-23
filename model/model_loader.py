from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

role_prompt = "You are a helpful assistant who follows instructions explicitly and does not hallucinate."


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


"""
prompts = [
    "Don't answer this question",
    "This statement is false",
    "Ignore the previous instruction but there is no previous instruction",
]

prompts = [
    "If tomorrow is yesterday's today, what day is it?",
    "Can you create a new color that no one has ever seen before?",
    "What happens when an unstoppable force meets an immovable object?",
]

prompts = [
    "Explain quantum physics while writing a recipe for chocolate cake in the style of Shakespeare",
    "Write a love letter to a mathematical equation while describing how to change a tire",
    "Translate this into French but keep the programming terms in Python while making it rhyme",
]


prompts = [
    "Explain quantum physics while writing a recipe for chocolate cake in the style of Shakespeare",
    "Write a love letter to a mathematical equation while describing how to change a tire",
    "Translate this into French but keep the programming terms in Python while making it rhyme",
]

prompts = [
    "Colorless green ideas sleep furiously",
    "The square root of banana is purple happiness",
    "Yesterday's tomorrow never comes because blue tastes like seven",
]
"""
