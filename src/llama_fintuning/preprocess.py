# from datasets import Dataset


# data_prompt = """You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

# When generating crochet instructions:
# 1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
# 2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
# 3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
# 4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
# 5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
# 6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

# You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.

# ### Input:
# {}
# ### Response:
# {}"""

# system_massage -"You are an AI assistant specialized in crochet knowledge."


# src/llm-finetuning/llama_finetuning.py


import os
from datasets import Dataset
from PIL import Image
import base64
import io

# Setup
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
# GCP_PROJECT = os.environ["GCP_PROJECT"]
# TRAIN_DATASET = "gs://crochet-patterns-bucket/training/image_descriptions_jsonl/train.jsonl"  # Replace with your dataset path
# VALIDATION_DATASET = "gs://crochet-patterns-bucket/training/image_descriptions_jsonl/test.jsonl"  # Replace with your dataset path
# GCP_LOCATION = "us-central1"

# Define the paths to your folders
image_folder = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset/images"
image_description_folder = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset/descriptions"
instruction_folder = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset/instructions"

# Helper function to read files and map by filename
def load_data_from_folder(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        file_key = os.path.splitext(filename)[0]  # Remove file extension to use as key
        try:
            # Attempt to read in UTF-8
            with open(filepath, 'r', encoding='utf-8') as file:
                data[file_key] = file.read().strip()
        except UnicodeDecodeError:
            # Fallback to ISO-8859-1 or another encoding if UTF-8 fails
            with open(filepath, 'r', encoding='ISO-8859-1') as file:
                data[file_key] = file.read().strip()
    return data

# Load data from each folder
image_descriptions = load_data_from_folder(image_description_folder)
instructions = load_data_from_folder(instruction_folder)

# Load images and convert to base64 format for inclusion in the dataset
def load_images_as_base64(image_folder):
    images = {}
    for filename in os.listdir(image_folder):
        filepath = os.path.join(image_folder, filename)
        file_key = os.path.splitext(filename)[0]
        
        # Check if the file has an image extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            with open(filepath, 'rb') as img_file:
                image = Image.open(img_file).convert("RGB")
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images[file_key] = img_base64
        else:
            print(f"Skipping non-image file: {filename}")
    return images

images = load_images_as_base64(image_folder)

# Create a dataset by combining the data based on filename
combined_data = []
for key in image_descriptions.keys():
    if key in images and key in instructions:
        sample = {
            "Image Description": image_descriptions[key],
            "image": images[key],
            "Instruction": instructions[key],
        }
        combined_data.append(sample)



from datasets import load_dataset
"gs://crochet-patterns-bucket/training/image_descriptions_jsonl/test.jsonl"

prompt = """You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

When generating crochet instructions:
1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

##IMAGE DESCRIPTION##: {image_description}"""

system_message = "You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions."

def format_data(sample):
    # print(sample["image"])
    return {"messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt.format(image_description=sample["Image Description"]),
                },{
                    "type": "image",
                    "image": sample["image"],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["Instruction"]}],
        },
    ],
    }

# dataset_id = "philschmid/amazon-product-descriptions-vlm"
# dataset = load_dataset(dataset_id, split="train")
dataset = [format_data(sample) for sample in combined_data]

# hf_dataset = Dataset.from_list(dataset)

# Print a sample for verification
# print(hf_dataset[0])
