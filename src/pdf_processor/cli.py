# pdf_processor.py
import os
import glob
import shutil
import pdfplumber
import cv2
import numpy as np
from google.cloud import storage, vision
import json
import time

# GCP configurations
bucket_name = os.environ["GCS_BUCKET_NAME"]
input_files = "input_files"
images_folder = "training/images"
txt_outputs = "training/text_instructions/txt_outputs"

def makedirs():
    os.makedirs(input_files, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(txt_outputs, exist_ok=True)

def upload_to_gcs(local_file, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file, timeout=600, num_retries=5)
    print(f"Uploaded {local_file} to {destination_blob_name}.")

def download():
    print("Downloading PDFs from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    folders = ["Afghans+%26+Blankets", "Pillows+%26+Poufs", "Rug", "Scarves", "Skirts", "Socks", "Tops"]
    # folders = ["Skirts"]

    for folder in folders:
        blobs = bucket.list_blobs(prefix=folder)
        for blob in blobs:
            if blob.name.endswith(".pdf"):
                local_path = os.path.join(input_files, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
                print(f"Downloaded {blob.name} to {local_path}")

def extract_text_from_pdf_gcs(gcs_uri, output_uri):
    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    gcs_source = vision.GcsSource(uri=gcs_uri)
    input_config = vision.InputConfig(gcs_source=gcs_source, mime_type="application/pdf")
    output_config = vision.OutputConfig(gcs_destination=vision.GcsDestination(uri=output_uri), batch_size=1)
    request = vision.AsyncAnnotateFileRequest(features=[feature], input_config=input_config, output_config=output_config)
    
    operation = client.async_batch_annotate_files(requests=[request])
    operation.result(timeout=300)
    print("Text extraction completed.")

def download_results_from_gcs(prefix, local_file):
    """Download the extracted text results from GCS and save only the text."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    full_text = ""  # Use to store text

    for blob in blobs:
        json_data = json.loads(blob.download_as_text())
        for response in json_data.get("responses", []):
            full_text += response.get("fullTextAnnotation", {}).get("text", "")

    with open(local_file, "w") as f:
        f.write(full_text)

    print(f"Results saved to {local_file}")

def extract_largest_image(pdf_path, output_image_path):
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        images = first_page.images
        if not images:
            print("No images found on the first page.")
            return
        largest_image = max(images, key=lambda img: (img["x1"] - img["x0"]) * (img["bottom"] - img["top"]))

        # Extract image coordinates
        x0, y0, x1, y1 = largest_image["x0"], largest_image["top"], largest_image["x1"], largest_image["bottom"]

        # Get page dimensions (bounding box)
        page_width = first_page.width
        page_height = first_page.height

        # Clamp the image bounding box within the page bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(page_width, x1)
        y1 = min(page_height, y1)

        # Ensure the bounding box is within the page
        if x1 <= x0 or y1 <= y0:
            print(f"Invalid bounding box: ({x0}, {y0}, {x1}, {y1})")
            return

        # Crop and save the image
        cropped_img = first_page.within_bbox((x0, y0, x1, y1)).to_image().original
        output_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, output_img)
        print(f"Largest image saved as {output_image_path}")

def upload_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    upload_to_gcs(pdf_path, f"{input_files}/{pdf_name}.pdf")

def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    gcs_pdf_uri = f"gs://{bucket_name}/input_files/{pdf_name}.pdf"
    json_output_uri = f"gs://{bucket_name}/training/text_instructions/json_outputs/{pdf_name}_output/"
    extract_text_from_pdf_gcs(gcs_pdf_uri, json_output_uri)

    text_file_path = os.path.join(txt_outputs, f"{pdf_name}.txt")
    download_results_from_gcs(f"training/text_instructions/json_outputs/{pdf_name}_output", text_file_path)
    
    image_path = os.path.join(images_folder, f"{pdf_name}.png")
    extract_largest_image(pdf_path, image_path)

def upload():
    """Upload all processed files to GCS."""
    for folder, gcs_folder in [
        (images_folder, "training/images"),
        (txt_outputs, "training/text_instructions/txt_outputs")
    ]:
        for local_file in glob.glob(os.path.join(folder, "*")):
            upload_to_gcs(local_file, f"{gcs_folder}/{os.path.basename(local_file)}")

def main(args):
    makedirs()
    if args.download:
        download()
    if args.uploadpdfs:
        for pdf_file in glob.glob(os.path.join(input_files, "*.pdf")):
            upload_pdf(pdf_file)
    if args.process:
        cnt = 0
        for pdf_file in glob.glob(os.path.join(input_files, "*.pdf")):
            pdf_name = os.path.basename(pdf_file).replace(".pdf", "")
            image_path = os.path.join(images_folder, f"{pdf_name}.png")
            if os.path.exists(image_path):
                print(f"{pdf_name}.pdf already processed, skip!")
                continue
            process_pdf(pdf_file)
            cnt += 1
            if cnt % 200 == 0:
                time.sleep(600)
            print("processed pdf cnt", cnt)
    if args.upload:
        upload()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process PDFs")
    parser.add_argument("-d", "--download", action="store_true", help="Download PDFs")
    parser.add_argument("-up", "--uploadpdfs", action="store_true", help="Upload PDFs to one folder")
    parser.add_argument("-p", "--process", action="store_true", help="Process PDFs")
    parser.add_argument("-u", "--upload", action="store_true", help="Upload files")
    args = parser.parse_args()
    main(args)
