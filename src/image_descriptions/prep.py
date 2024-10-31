import os
import glob
import argparse
import json
import random
from PIL import Image as PILImage
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
import time

base_folder = "training"

images_folder = f"{base_folder}/images"
text_instructions_folder = f"{base_folder}/text_instructions/txt_outputs"
image_descriptions_txt_folder = f"{base_folder}/image_descriptions_txt"
image_descriptions_json_llama_folder = f"{base_folder}/image_descriptions_json_llama"

os.makedirs(image_descriptions_json_llama_folder, exist_ok=True)

def create_json_file(image_descriptions_txt_folder, text_instructions_folder, image_descriptions_json_llama_folder):
    """Create a JSON file with structures that can be used to finetune llama later."""
    # Get the list of files from both folders
    image_description_files = os.listdir(image_descriptions_txt_folder)
    text_instruction_files = os.listdir(text_instructions_folder)
    
    for image_description_file in image_description_files:
        # Extract the base name (without extension) to match with text instruction files
        image_name, _ = os.path.splitext(image_description_file)
        
        # Find the matching text instruction file
        matching_text_instruction_file = f"{image_name}.txt"
        
        if matching_text_instruction_file in text_instruction_files:
            # Paths to the files
            image_description_path = os.path.join(image_descriptions_txt_folder, image_description_file)
            text_instruction_path = os.path.join(text_instructions_folder, matching_text_instruction_file)
            
            # Read the content of the image description file
            with open(image_description_path, "r") as img_desc_file:
                image_descriptions = img_desc_file.read().strip()
            
            # Read the content of the text instruction file
            with open(text_instruction_path, "r") as txt_instr_file:
                text_instruction = txt_instr_file.read().strip()
            
            # Construct the JSON content
            json_content = {
                "context": image_descriptions,
                "response": text_instruction
            }
            
            # Create a JSON file path based on the image name
            json_file_path = os.path.join(image_descriptions_json_llama_folder, f"{image_name}.json")
            
            # Write the JSON content to the file
            with open(json_file_path, "w") as json_file:
                json.dump(json_content, json_file, indent=4)

            print(f"JSON file created for {image_name} at {json_file_path}")