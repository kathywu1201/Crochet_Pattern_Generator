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

# dataset_folder = os.path.join("/persistent", "dataset")
# raw_image_folder = os.path.join(dataset_folder, "raw_image")
# raw_instructions_folder = os.path.join(dataset_folder, "raw_instructions/txt_outputs")
# image_descriptions_txt_folder = os.path.join(dataset_folder, "/image_descriptions_txt")
# image_descriptions_json_folder = os.path.join(dataset_folder,"/image_descriptions_json")
# image_descriptions_jsonl_folder = os.path.join(dataset_folder,"/image_descriptions_jsonl")
# cleaned_text_instructions_folder = os.path.join(dataset_folder,"/cleaned_text_instructions")

gcp_project = os.environ["GCP_PROJECT"]
bucket_name = os.environ["GCS_BUCKET_NAME"]
region = "us-central1"

base_folder = "training"
raw_image_folder = f"{base_folder}/images"
raw_instructions_folder = f"{base_folder}/text_instructions/txt_outputs"
image_descriptions_txt_folder = f"{base_folder}/image_descriptions_txt"
image_descriptions_json_folder = f"{base_folder}/image_descriptions_json"
image_descriptions_jsonl_folder = f"{base_folder}/image_descriptions_jsonl"
cleaned_text_instructions_folder = f"{base_folder}/cleaned_text_instructions"


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
    os.makedirs(raw_image_folder, exist_ok=True)
    os.makedirs(raw_instructions_folder, exist_ok=True)
    os.makedirs(cleaned_text_instructions_folder, exist_ok=True)


def upload_to_gcs(bucket_name, local_file, destination_blob_name):
    """Upload a local file to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to {destination_blob_name}.")

def download_files_from_gcs(bucket_name, prefix, local_folder):
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

def clean_instructions(max_retries=5, retry_delay=5):
    """Clean text instructions using Gemini and save the cleaned output with retry logic."""
    print("Cleaning text instructions...")

    # Get all txt files in the text_instructions_folder
    txt_files = [f for f in os.listdir(raw_instructions_folder) if f.endswith('.txt')]

    if not txt_files:
        print(f"No instruction files found in '{raw_instructions_folder}'.")
        return

    for txt_file in txt_files:
        txt_file_path = os.path.join(raw_instructions_folder, txt_file)

        # Read the content of the text file
        with open(txt_file_path, 'r') as file:
            raw_text = file.read()

        # Define the prompt for Gemini
        cleaning_prompt = f"""
        You are an expert in data extraction and text summarization with a deep understanding of crochet instructions. Your task is to analyze the provided text file containing raw crochet instructions, which may include noisy background information, and extract the following key details:

        1. **Crochet Product Name:** The specific name or title of the crochet project.
        2. **Materials:** A list of materials needed for the project, including details such as yarn type, hook size, and other required items.
        3. **Abbreviations:** All abbreviations mentioned in the text, along with their meanings (e.g., SC: Single Crochet, DC: Double Crochet).
        4. **Measurements:** The dimensions or sizes relevant to the crochet project.
        5. **Instructions:** The step-by-step directions for creating the crochet item.

        Please ignore any irrelevant or background information and focus exclusively on the above sections. Organize the extracted details in the following structured format for clarity and consistency:

        ---
        **Cleaned Text Format:**

        **Crochet Product Name:**  
        [Insert name of the product]

        **Materials:**  
        - [Material 1: Description]  
        - [Material 2: Description]  
        - [Material 3: Description]

        **Abbreviations:**  
        - [Abbreviation 1: Meaning]  
        - [Abbreviation 2: Meaning]  
        - [Abbreviation 3: Meaning]

        **Measurements:**  
        [Insert dimensions, sizes, or any measurement-related details]

        **Instructions:**  
        1. [Step 1: Description]  
        2. [Step 2: Description]  
        3. [Step 3: Description]

        ---
        Additional Guidelines:
        1. Ensure clarity and conciseness while extracting and organizing the information.
        2. Use proper crochet terminology for technical accuracy.
        3. Retain the sequence and structure of the instructions as much as possible.

        If any section is missing or not explicitly mentioned in the text file, state "Not Available" for that section.

        Below is the raw text content:
        ---
        {raw_text}
        """

        # Retry logic
        retry_count = 0
        cleaned_text = None

        while retry_count < max_retries:
            try:
                # Initialize Gemini model
                generative_model = GenerativeModel("gemini-1.5-pro-002")
                response = generative_model.generate_content([cleaning_prompt])
                cleaned_text = response.text if hasattr(response, 'text') else "Error: Unable to generate cleaned instructions."
                break  # Exit the loop if processing is successful

            except Exception as e:
                if '429' in str(e):  # Check if the error is a quota-related issue (HTTP 429)
                    retry_count += 1
                    print(f"Quota exceeded for {txt_file}. Retrying in {retry_delay} seconds... (Retry {retry_count}/{max_retries})")
                    time.sleep(retry_delay)  # Wait for the specified time before retrying
                else:
                    print(f"Error cleaning instructions for {txt_file}: {e}")
                    cleaned_text = "Error: Unable to process this file."
                    break  # Exit loop if it's an unexpected error

        # If retries are exhausted and still no result
        if cleaned_text is None:
            print(f"Failed to clean instructions for {txt_file} after {max_retries} retries.")
            cleaned_text = "Failed to generate cleaned instructions after multiple retries."

        # Save the cleaned content to the new folder with the original name
        cleaned_file_path = os.path.join(cleaned_text_instructions_folder, txt_file)
        with open(cleaned_file_path, 'w') as cleaned_file:
            cleaned_file.write(cleaned_text)

        print(f"Cleaned instructions saved for {txt_file}: {cleaned_file_path}")

    print("All text instructions cleaned.")

def process():
    """Generate descriptions for all images, load text instructions, and save as JSON and TXT."""
    image_files = [f for f in os.listdir(raw_image_folder) if f.endswith(".png")]

    if not image_files:
        print(f"No images found in '{raw_image_folder}'.")
        return

    image_processed_cnt = 0
    excluded_image_cnt = 0

    for image_file in image_files:
        descriptions_file_path = os.path.join(image_descriptions_txt_folder, f"{os.path.splitext(image_file)[0]}.txt")
        if os.path.exists(descriptions_file_path):
            print(f"{image_file} description already processed.")
            continue

        image_path = os.path.join(raw_image_folder, image_file)
        description = generate_image_description(image_path)
        if "Error generating description" in description or "not crochet" in description or "*not* crochet" in description:
            print(f"{image_file} excluded from dataset.")
            excluded_image_cnt += 1
            continue 

        save_txt_file(image_file, description)

        txt_file_path = os.path.join(cleaned_text_instructions_folder, f"{os.path.splitext(image_file)[0]}.txt")
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

def generate_json_from_existing_files():
    """
    Create JSON files for each image description in the folder.
    If a corresponding cleaned text instruction exists, use it to generate JSON.
    """
    print("Generating JSON files from existing descriptions and cleaned instructions...")

    # List all image description text files
    image_description_files = [f for f in os.listdir(image_descriptions_txt_folder) if f.endswith('.txt')]
    if not image_description_files:
        print(f"No image description files found in '{image_descriptions_txt_folder}'.")
        return

    for description_file in image_description_files:
        image_name = os.path.splitext(description_file)[0]
        description_path = os.path.join(image_descriptions_txt_folder, description_file)

        # Read the image description
        with open(description_path, 'r') as desc_file:
            image_description = desc_file.read().strip()

        # Check if a corresponding cleaned instruction exists
        cleaned_instruction_path = os.path.join(cleaned_text_instructions_folder, f"{image_name}.txt")
        if os.path.exists(cleaned_instruction_path):
            with open(cleaned_instruction_path, 'r') as instr_file:
                cleaned_instruction = instr_file.read().strip()
        else:
            cleaned_instruction = "No corresponding cleaned instruction available."
            print(f"No cleaned instruction found for {image_name}.txt")

        # Create the JSON file
        create_json_file(image_name, image_description, cleaned_instruction)

    print("Finished generating JSON files.")

def split_json_to_jsonl(input_folder, output_folder, train_ratio=0.93, val_ratio=0.07, test_ratio=0):
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
    for local_file in glob.glob(os.path.join(cleaned_text_instructions_folder, "*.jsonl")):
        upload_to_gcs(local_file, f"{base_folder}/cleaned_text_instructions/{os.path.basename(local_file)}")


def main(args):
    makedirs()

    # bucket_name = os.getenv("GCS_BUCKET_NAME")
    if args.bucket != "":
        bucket_name = args.bucket
    print(f"Using GCS Bucket: {bucket_name}")

    if args.download:
        download_files_from_gcs(bucket_name, "training/images/", raw_image_folder)
        download_files_from_gcs(bucket_name, "training/text_instructions/txt_outputs/", raw_instructions_folder)
    if args.process:
        process()
    if args.clean_instructions:
        clean_instructions()
    if args.generate_json:
        generate_json_from_existing_files()
    if args.split:
        split_json_to_jsonl(image_descriptions_json_folder, image_descriptions_jsonl_folder)
    if args.upload:
        upload(bucket_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image descriptions and upload to GCS.")
    parser.add_argument("-d", "--download", action="store_true", help="Download images and text instructions from GCS")
    parser.add_argument("-ci", "--clean_instructions", action="store_true", help="Clean instructions and save as TXT files")
    parser.add_argument("-p", "--process", action="store_true", help="Process images and generate TXT and JSON files")
    parser.add_argument("-s", "--split", action="store_true", help="Split JSON files into train/val/test JSONL files")
    parser.add_argument("-u", "--upload", action="store_true", help="Upload all files to GCS")
    parser.add_argument("-b", "--bucket", type=str, default = "", help="GCS Bucket Name to override the environment variable")
    parser.add_argument("-gj", "--generate_json", action="store_true", help="Generate JSON files from existing descriptions and cleaned instructions (only use if image descriptions already exist)")
    args = parser.parse_args()
    main(args)