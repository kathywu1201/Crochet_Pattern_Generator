import os
import io
import glob
import torch
import shutil
import numpy as np
import pdfplumber
import cv2
import json
from PIL import Image
from transformers import AutoFeatureExtractor, Swinv2Model
from google.cloud import storage, vision
import argparse

# GCP configurations
gcp_project = "crochetAI"
bucket_name = "crochet-patterns-bucket"
base_folder = "training"  
input_files = f"input_files"
images_folder = f"{base_folder}/images"
image_vectors = f"{base_folder}/image_vectors"
text_instructions = f"{base_folder}/text_instructions"
# json_outputs = f"{text_instructions}/json_outputs"
txt_outputs = f"{text_instructions}/txt_outputs"

def makedirs():
    """Create necessary local directories."""
    os.makedirs(input_files, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(image_vectors, exist_ok=True)
    os.makedirs(txt_outputs, exist_ok=True)

def upload_to_gcs(local_file, destination_blob_name):
    """Upload a local file to GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to {destination_blob_name}.")

def download():
    """Download files from 'input_files' folder on GCS."""
    print("Downloading files from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    folders = ["Pillows+%26+Poufs", "Scarves", "Skirts", "Afghans+%26+Blankets", "Rug"]

    # Loop through each folder and download PDFs
    for folder in folders:
        print(f"Downloading from folder: {folder}")
        blobs = bucket.list_blobs(prefix=folder)

        for blob in blobs:
            if blob.name.endswith(".pdf"):
                local_path = os.path.join(input_files, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
                print(f"Downloaded {blob.name} to {local_path}")

def extract_text_from_pdf_gcs(gcs_uri, output_uri):
    """Extract text from PDF using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision.GcsSource(uri=gcs_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    gcs_destination = vision.GcsDestination(uri=output_uri)
    output_config = vision.OutputConfig(gcs_destination=gcs_destination, batch_size=1)

    request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config
    )

    print("Waiting for Vision API to process the PDF...")
    operation = client.async_batch_annotate_files(requests=[request])
    operation.result(timeout=300)
    print("Text extraction completed.")

def download_results_from_gcs(prefix, local_file):
    """Download the extracted text results from GCS and save only the text."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    full_text = ""  # Initialize an empty string to store the extracted text

    for blob in blobs:
        # Load the JSON response from each blob
        json_data = json.loads(blob.download_as_text())
        # Extract text from the Vision API's response
        for response in json_data.get("responses", []):
            full_text += response.get("fullTextAnnotation", {}).get("text", "")

    # Save the extracted text to a local .txt file
    with open(local_file, "w") as f:
        f.write(full_text)

    print(f"Results saved to {local_file}")

def extract_and_auto_crop_color(pdf_path, output_image_path):
    """Extract and crop the first image from a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        page_image = first_page.to_image().original

        temp_path = "temp_full_page.png"
        page_image.save(temp_path)

    img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    cv2.imwrite(output_image_path, cropped_img)
    print(f"Cropped color image saved as {output_image_path}")

def image_to_vector(image_path):
    """Convert an image to a vector using SwinV2."""
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    model = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window8-256")

    image = Image.open(image_path).convert("RGB")
    inputs = extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        vector = outputs.last_hidden_state.mean(dim=1).squeeze()

    return vector.numpy()

def process_pdf(pdf_path):
    """Process a PDF: Extract text, crop images, and generate vectors."""
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    upload_to_gcs(pdf_path, f"{input_files}/{pdf_name}.pdf")

    gcs_pdf_uri = f"gs://{bucket_name}/input_files/{pdf_name}.pdf"
    json_output_uri = f"gs://{bucket_name}/training/text_instructions/json_outputs/{pdf_name}_output/"
    extract_text_from_pdf_gcs(gcs_pdf_uri, json_output_uri)

    text_file_path = os.path.join(txt_outputs, f"{pdf_name}.txt")
    download_results_from_gcs(f"training/text_instructions/json_outputs/{pdf_name}_output", text_file_path)

    image_path = os.path.join(images_folder, f"{pdf_name}.png")
    extract_and_auto_crop_color(pdf_path, image_path)

    if os.path.exists(image_path):
        vector = image_to_vector(image_path)
        vector_path = os.path.join(image_vectors, f"{pdf_name}.npy")
        np.save(vector_path, vector)

def upload():
    """Upload all processed files to GCS."""
    for folder, gcs_folder in [
        (images_folder, "training/images"),
        (image_vectors, "training/image_vectors"),
        (txt_outputs, "training/text_instructions/txt_outputs")
    ]:
        for local_file in glob.glob(os.path.join(folder, "*")):
            upload_to_gcs(local_file, f"{gcs_folder}/{os.path.basename(local_file)}")

def main(args):
    makedirs()

    if args.download:
        download()
    if args.process:
        pdf_files = [f for f in os.listdir(input_files) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_files, pdf_file)
            process_pdf(pdf_path)
    if args.upload:
        upload()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDFs and organize results.")
    parser.add_argument("-d", "--download", action="store_true", help="Download PDFs from GCS")
    parser.add_argument("-p", "--process", action="store_true", help="Process PDFs")
    parser.add_argument("-u", "--upload", action="store_true", help="Upload processed files to GCS")
    args = parser.parse_args()
    main(args)