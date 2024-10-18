import os
import glob
import argparse
import json
import random
from PIL import Image as PILImage
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image

# GCP configurations
gcp_project = "crochetai-438515"
region = "us-central1"
bucket_name = "crochet-patterns-bucket"
base_folder = "training"
images_folder = f"{base_folder}/images"
text_instructions_folder = f"{base_folder}/text_instructions/txt_outputs"
image_descriptions_txt_folder = f"{base_folder}/image_descriptions_txt"
image_descriptions_json_folder = f"{base_folder}/image_descriptions_json"
image_descriptions_jsonl_folder = f"{base_folder}/image_descriptions_jsonl"

# Initialize Vertex AI
vertexai.init(project=gcp_project, location=region)

def makedirs():
    os.makedirs(image_descriptions_txt_folder, exist_ok=True)
    os.makedirs(image_descriptions_json_folder, exist_ok=True)
    os.makedirs(image_descriptions_jsonl_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(text_instructions_folder, exist_ok=True)

def upload_to_gcs(local_file, destination_blob_name):
    """Upload a local file to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to {destination_blob_name}.")

def download_files_from_gcs(prefix, local_folder):
    """Download files from a specific GCS folder to a local directory."""
    print(f"Downloading files from '{prefix}'...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        local_path = os.path.join(local_folder, os.path.basename(blob.name))
        if not os.path.basename(blob.name):
            continue  # Skip directories or empty paths
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

def generate_image_description(image_path):
    """Generate a caption for an image using the Gemini model."""
    print(f"Generating detailed description for {image_path}...")

    image = Image.load_from_file(image_path)
    generative_model = GenerativeModel("gemini-1.5-pro-002")
    response = generative_model.generate_content(
        ["Describe the crochet product shown in this image, focusing on its pattern and texture. Ignore any other objects or elements in the image.", image]
    )

    description = response.text
    print(f"Generated description: {description}")
    return description

def save_txt_file(image_name, description):
    """Save the description as a text file."""
    txt_file_path = os.path.join(image_descriptions_txt_folder, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_file_path, "w") as txt_file:
        txt_file.write(description)
    print(f"Saved TXT file: {txt_file_path}")

def create_json_file(image_name, user_text, model_text):
    """Create a JSON file with structures that can be used to finetune gemini later."""
    json_content = {
        "contents": [
            {"role": "user", "parts": [{"text": user_text}]},
            {"role": "model", "parts": [{"text": model_text}]}
        ]
    }

    json_file_path = os.path.join(image_descriptions_json_folder, f"{os.path.splitext(image_name)[0]}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_content, json_file, indent=4)
    print(f"Saved JSON file: {json_file_path}")

def process():
    """Generate descriptions for all images, load text instructions, and save as JSON and TXT."""
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]

    if not image_files:
        print(f"No images found in '{images_folder}'.")
        return

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        description = generate_image_description(image_path)
        save_txt_file(image_file, description)

        txt_file_path = os.path.join(text_instructions_folder, f"{os.path.splitext(image_file)[0]}.txt")
        if os.path.exists(txt_file_path):
            with open(txt_file_path, "r") as txt_file:
                text_instruction = txt_file.read().strip()
        else:
            text_instruction = "No text instruction available."
            print(f"Warning: No corresponding text file for {image_file}")

        create_json_file(image_file, description, text_instruction)

def split_json_to_jsonl(input_folder, output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split JSON files into train, validation, and test JSONL files."""
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    random.shuffle(json_files)

    total_files = len(json_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    train_files = json_files[:train_count]
    val_files = json_files[train_count:train_count + val_count]
    test_files = json_files[train_count + val_count:]

    os.makedirs(output_folder, exist_ok=True)

    def write_to_jsonl(files, jsonl_filename):
        with open(jsonl_filename, 'w') as jsonl_file:
            for json_file in files:
                with open(os.path.join(input_folder, json_file), 'r') as f:
                    data = json.load(f)
                    jsonl_file.write(json.dumps(data) + '\n')
        print(f"{len(files)} files written to {jsonl_filename}")

    write_to_jsonl(train_files, os.path.join(output_folder, "train.jsonl"))
    write_to_jsonl(val_files, os.path.join(output_folder, "validation.jsonl"))
    write_to_jsonl(test_files, os.path.join(output_folder, "test.jsonl"))

def upload():
    """Upload all generated files to GCS."""
    for local_file in glob.glob(os.path.join(image_descriptions_txt_folder, "*.txt")):
        upload_to_gcs(local_file, f"{base_folder}/image_descriptions_txt/{os.path.basename(local_file)}")
    for local_file in glob.glob(os.path.join(image_descriptions_json_folder, "*.json")):
        upload_to_gcs(local_file, f"{base_folder}/image_descriptions_json/{os.path.basename(local_file)}")
    for local_file in glob.glob(os.path.join(image_descriptions_jsonl_folder, "*.jsonl")):
        upload_to_gcs(local_file, f"{base_folder}/image_descriptions_jsonl/{os.path.basename(local_file)}")

def main(args):
    makedirs()

    if args.download:
        download_files_from_gcs("training/images/", images_folder)
        download_files_from_gcs("training/text_instructions/txt_outputs/", text_instructions_folder)
    if args.process:
        process()
    if args.split:
        split_json_to_jsonl(image_descriptions_json_folder, image_descriptions_jsonl_folder)
    if args.upload:
        upload()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image descriptions and upload to GCS.")
    parser.add_argument("-d", "--download", action="store_true", help="Download images and text instructions from GCS")
    parser.add_argument("-p", "--process", action="store_true", help="Process images and generate TXT and JSON files")
    parser.add_argument("-s", "--split", action="store_true", help="Split JSON files into train/val/test JSONL files")
    parser.add_argument("-u", "--upload", action="store_true", help="Upload all files to GCS")
    args = parser.parse_args()
    main(args)
