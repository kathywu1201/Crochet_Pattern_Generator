import os
from google.cloud import storage

def upload_folder_to_gcs(bucket_name, source_folder, destination_blob_prefix=""):
    """Uploads an entire folder to GCS."""
    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Walk through all files and directories in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            # Full local file path
            local_file_path = os.path.join(root, file_name)

            # Create the blob path (substitute slashes for GCS-style paths)
            relative_path = os.path.relpath(local_file_path, source_folder)
            blob_name = os.path.join(destination_blob_prefix, relative_path).replace("\\", "/")

            # Create a blob object and upload the file
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)

            print(f"Uploaded {local_file_path} to {blob_name}")

# Example usage
bucket_name = "crochet-patterns-bucket"
source_folder = "/Users/ciciwxp/Desktop/AC215/input_file"
upload_folder_to_gcs(bucket_name, source_folder)
