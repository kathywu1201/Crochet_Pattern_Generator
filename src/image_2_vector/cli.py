# vector_processor.py
import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, Swinv2Model
from google.cloud import storage

# GCP configurations
bucket_name = os.environ["GCS_BUCKET_NAME"]

# Training use this
# images_folder = "training/images"
# image_vectors = "training/image_vectors"

# User use this
images_folder = "user/images"
image_vectors = "user/image_vectors"

def makedirs():
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(image_vectors, exist_ok=True)

def upload_to_gcs(local_file, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to {destination_blob_name}.")

def download_images():
    print("Downloading images from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix="training/images")

    for blob in blobs:
        if blob.name.endswith(".png"):
            local_path = os.path.join(images_folder, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")

def image_to_vector(image_path):
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    model = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def process_images():
    cnt = 0
    for image_file in glob.glob(os.path.join(images_folder, "*.png")):
        vector_path = os.path.join(image_vectors, f"{os.path.basename(image_file).replace('.png', '.npy')}")

        if os.path.exists(vector_path):
            print(f"Vector for {image_file} already exists. Skipping.")
            continue
        
        vector = image_to_vector(image_file)
        vector_path = os.path.join(image_vectors, f"{os.path.basename(image_file).replace('.png', '.npy')}")
        np.save(vector_path, vector)
        cnt += 1
        if cnt%20==0:
            print("processed image cnt", cnt)

def upload_vectors():
    for vector_file in glob.glob(os.path.join(image_vectors, "*.npy")):
        upload_to_gcs(vector_file, f"training/image_vectors/{os.path.basename(vector_file)}")

def main(args):
    makedirs()
    if args.download:
        download_images()
    if args.process:
        process_images()
    if args.upload:
        upload_vectors()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Images")
    parser.add_argument("-d", "--download", action="store_true", help="Download images")
    parser.add_argument("-p", "--process", action="store_true", help="Process images")
    parser.add_argument("-u", "--upload", action="store_true", help="Upload vectors")
    args = parser.parse_args()
    main(args)
