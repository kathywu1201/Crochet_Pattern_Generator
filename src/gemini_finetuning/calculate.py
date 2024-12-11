import json
from vertexai.generative_models import GenerativeModel

# Initialize the GenerativeModel
model = GenerativeModel("models/gemini-1.5-flash")

# Path to the JSONL file
jsonl_file_path = "image_descriptions_jsonl/train.jsonl"

# Function to calculate token count of text parts in JSONL
def calculate_text_token_count(jsonl_file_path, model):
    total_tokens = 0
    line_count = 0

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            line_count += 1
            data = json.loads(line.strip())
            contents = data.get("contents", [])
            
            for content in contents:
                parts = content.get("parts", [])
                for part in parts:
                    text = part.get("text", "")
                    if text:
                        # Count tokens for the text
                        response = model.count_tokens(text)
                        text_tokens = response.total_tokens  # Extract the token count
                        total_tokens += text_tokens

    return {"total_tokens": total_tokens, "line_count": line_count}

# Example usage
result = calculate_text_token_count(jsonl_file_path, model)
print("Token Count Summary:", result)