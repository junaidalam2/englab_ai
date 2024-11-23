# LLM AI

## Application Structure

- The structure includes flask application for API handling, LLM Model (gpt2)
- Accept user prompts related to spiritual guidance.
- Use the LLM to generate personalized insights or practices based on ancient teachings.
- Present these responses in a user-friendly format.


## Prerequisites

- Python 3.12+
- flask
- transformers
- torch
- Access to GitHub Models (for GITHUB_TOKEN)
- SQLite


## Setup Instructions

1. Clone the repository:

git clone https://github.com/junaidalam2/englab_ai.git

2. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```


4. Start the development server:

```bash
python3 app.py
```

