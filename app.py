from flask import Flask, request, jsonify
from model import generate_response


app = Flask(__name__)


@app.route("/", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    response = generate_response(prompt)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
