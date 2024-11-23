from flask import Flask, request, jsonify
from model import generate_response, scriptures, meditation_tips
from random import choice

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    # Get the user prompt from the request
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Randomly select a scripture or meditation tip for context
    context = choice(scriptures + meditation_tips)
    enhanced_prompt = f"{context}\n{prompt}"

    # Generate the response
    response = generate_response(enhanced_prompt)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
