import os
import logging
import json
from datasets import Dataset

# Paths
image_folder = "dataset/images"
# image_description_folder = "dataset/descriptions"
instruction_folder = "dataset/cleaned_text_instructions"
output_dataset_path = "dataset/filtered_dataset.json"

# Helper function to count tokens
def count_tokens(instruction):
    return len(instruction)

# Function to load data and filter based on token size
def load_data(max_tokens=20000):
    data = []
    excluded_count = 0  # Counter for excluded entries
    for filename in os.listdir(instruction_folder):
        base_name = os.path.splitext(filename)[0]
        # description_path = os.path.join(image_description_folder, filename)
        image_path = os.path.join(image_folder, f"{base_name}.png")  # Assuming images are .png
        instruction_path = os.path.join(instruction_folder, f"{base_name}.txt")

        if os.path.exists(image_path) and os.path.exists(instruction_path):
            try:
                with open(instruction_path, 'r') as f:
                    instruction = f.read().strip()
                if count_tokens(instruction) <= max_tokens:
                    data.append({"image_path": image_path, "instruction": instruction})
                else:
                    excluded_count += 1  # Increment counter for excluded entries
            except Exception as e:
                logging.error(f"Error reading files for '{base_name}': {e}")
                continue
    print(f"Total entries before filtering: {len(data) + excluded_count}")
    print(f"Total entries excluded due to large token size: {excluded_count}")
    return Dataset.from_list(data)

# Main execution
dataset = load_data(max_tokens=20000)

# Save the dataset as a JSON file
dataset_dict = dataset.to_dict()
with open(output_dataset_path, "w") as outfile:
    json.dump(dataset_dict, outfile)

print(f"Filtered dataset saved to {output_dataset_path} with {len(dataset)} entries.")

# To load the dataset later
def load_filtered_dataset(path):
    with open(path, "r") as infile:
        dataset_dict = json.load(infile)
    return Dataset.from_dict(dataset_dict)

# Example usage
loaded_dataset = load_filtered_dataset(output_dataset_path)
print(f"Loaded dataset contains {len(loaded_dataset)} entries.")
