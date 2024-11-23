from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load the configuration data
with open("config.json", "r") as config_file:
    config = json.load(config_file)

scriptures = config["scriptures"]
meditation_tips = config["meditation_tips"]

def generate_response(prompt):
    # Randomly select a scripture or meditation tip for context
    from random import choice
    context = choice(scriptures + meditation_tips)

    # prompt with selected context
    full_prompt = (
    f"As a spiritual guide rooted in ancient wisdom, provide actionable insights and "
    f"philosophical teachings to help someone with this question:\n\n"
    f"Context: {context}\n\n"
    f"Question: {prompt}"
)

    inputs = tokenizer(full_prompt, return_tensors="pt")

    # Generate text with adjusted parameters
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_return_sequences=1,
        temperature=0.8,  # Adds randomness
        top_p=0.9,       # Nucleus sampling
        top_k=50         # Limits to top 50 tokens
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()  # Strip trailing whitespace

