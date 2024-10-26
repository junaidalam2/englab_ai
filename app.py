from flask import Flask, request
import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

app = Flask(__name__)

@app.route("/",  methods=['POST'])
def promptAI():
    
    data = request.get_json()
    print(data)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        model=model_name
    )
    
    return response.choices[0].message.content


@app.route('/endpoint', methods=['POST'])
def handle_request():
    data = request.get_json()  # Get the JSON data from the request body
    # Process the data
    return 'Received data: {}'.format(data)