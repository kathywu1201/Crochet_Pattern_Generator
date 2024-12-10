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

# GCP configurations
gcp_project = os.environ["GCP_PROJECT"]
region = "us-central1"
bucket_name = os.environ["GCS_BUCKET_NAME"]
base_folder = "training"
images_folder = f"{base_folder}/images"
text_instructions_folder = f"{base_folder}/text_instructions/txt_outputs"
image_descriptions_txt_folder = f"{base_folder}/image_descriptions_txt"
image_descriptions_json_folder = f"{base_folder}/image_descriptions_json"
image_descriptions_jsonl_folder = f"{base_folder}/image_descriptions_jsonl"

# Define prompt
instruction = '''
You are an expert in textile arts with a specialization in crochet. 
Your task is to analyze the provided image of a crochet object and generate a detailed description focusing exclusively on the intricate details of the crochet work. 
Describe the crochet product shown in this image, focusing on its pattern and texture. 
Ignore any other objects or elements in the image.
The description should encompass the following aspects:

Number of Threads:
Determine and specify the total number of threads used in creating the crochet object.
If possible, provide information on the thickness or gauge of the threads.
Stitch Types:
Identify and list all the types of stitches present in the crochet piece (e.g., single crochet, double crochet, half-double crochet, treble crochet, etc.).
Describe any unique or complex stitch patterns utilized.
Yarn Color:
Accurately describe the color(s) of the yarn used.
Mention any color variations, gradients, or patterns resulting from the yarn colors.
Knit vs. Crochet Distinction:
Analyze the construction of the object to determine whether it is knitted or crocheted.
Provide reasoning for the distinction, highlighting specific features that indicate crochet techniques over knitting, or vice versa.
Number of Rows:
Count and state the total number of rows involved in the creation of the crochet product.
If applicable, mention any notable changes in row patterns or techniques throughout the project.
Additional Guidelines:

Focus Exclusively on Crochet Details: Ensure that the description remains concentrated on the crochet aspects mentioned above. Avoid general comments about the object's appearance, functionality, or aesthetics unless they directly relate to the crochet techniques or materials used.
Clarity and Precision: Use clear and precise language to convey each detail. Where measurements or counts are involved, provide them in appropriate units (e.g., number of threads, specific row counts).
Technical Accuracy: Ensure that all crochet terminology and descriptions are technically accurate, reflecting a deep understanding of crochet methods and practices.
Structured Format: Present the information in a well-organized manner, possibly using bullet points or numbered lists for each of the five key aspects to enhance readability.
'''

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

def generate_image_description(image_path, max_retries=5, retry_delay=5):
    """Generate a caption for an image using the Gemini model, with retry logic for 429 errors."""
    print(f"Generating detailed description for {image_path}...")

    generative_model = GenerativeModel("gemini-1.5-pro-002")
    retry_count = 0

    while retry_count < max_retries:
        try:
            image = Image.load_from_file(image_path)
            response = generative_model.generate_content([instruction, image])
            
            # Check if the response contains the expected text
            description = response.text if hasattr(response, 'text') else "No description available."
            print(f"Description generated for {image_path}.")
            return description

        except Exception as e:
            if '429' in str(e):  # Check if the error is a quota-related issue (HTTP 429)
                retry_count += 1
                print(f"Quota exceeded for {image_path}. Retrying in {retry_delay} seconds... (Retry {retry_count}/{max_retries})")
                time.sleep(retry_delay)  # Wait for the specified time before retrying
            else:
                print(f"Error generating description for {image_path}: {e}")
                return "Error generating description"

    # If all retries are exhausted, return a failure message
    print(f"Failed to generate description for {image_path} after {max_retries} retries.")
    return "Failed to generate description after multiple retries"

# def generate_image_description(image_path):
#     """Generate a caption for an image using the Gemini model."""
#     print(f"Generating detailed description for {image_path}...")

#     image = Image.load_from_file(image_path)
#     generative_model = GenerativeModel("gemini-1.5-pro-002")
#     response = generative_model.generate_content(
#         [instruction, image]
#     )

#     description = response.text
#     # print(f"Generated description: {description}")
#     return description

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

    image_processed_cnt = 0
    excluded_image_cnt = 0

    for image_file in image_files:
        descriptions_file_path = os.path.join(image_descriptions_txt_folder, f"{os.path.splitext(image_file)[0]}.txt")
        if os.path.exists(descriptions_file_path):
            print(f"{image_file} description already processed.")
            continue

        image_path = os.path.join(images_folder, image_file)
        description = generate_image_description(image_path)
        if "Error generating description" in description or "not crochet" in description or "*not* crochet" in description:
            print(f"{image_file} excluded from dataset.")
            excluded_image_cnt += 1
            continue 

        save_txt_file(image_file, description)

        txt_file_path = os.path.join(text_instructions_folder, f"{os.path.splitext(image_file)[0]}.txt")
        if os.path.exists(txt_file_path):
            with open(txt_file_path, "r") as txt_file:
                text_instruction = txt_file.read().strip()
        else:
            text_instruction = "No text instruction available."
            print(f"Warning: No corresponding text file for {image_file}")

        create_json_file(image_file, description, text_instruction)

        print("processed image cnt:", image_processed_cnt)
        image_processed_cnt += 1
    
    print("total processed images", image_processed_cnt)
    print("excluded image count", excluded_image_cnt)

def split_json_to_jsonl(input_folder, output_folder, train_ratio=0.85, val_ratio=0, test_ratio=0.15):
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
