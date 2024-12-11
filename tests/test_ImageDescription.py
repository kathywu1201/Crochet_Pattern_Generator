import os
import json
import pytest
import numpy as np
import shutil
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage
from image_descriptions.cli import (
    makedirs,
    upload_to_gcs,
    download_files_from_gcs,
    generate_image_description,
    save_txt_file,
    create_json_file,
    process,
    split_json_to_jsonl,
    upload
)

# Directory paths for test
TEST_BASE_FOLDER = "test_training"
TEST_IMAGES_FOLDER = f"{TEST_BASE_FOLDER}/images"
TEST_TEXT_INSTRUCTIONS_FOLDER = f"{TEST_BASE_FOLDER}/text_instructions/txt_outputs"
TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER = f"{TEST_BASE_FOLDER}/image_descriptions_txt"
TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER = f"{TEST_BASE_FOLDER}/image_descriptions_json"
TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER = f"{TEST_BASE_FOLDER}/image_descriptions_jsonl"
TEST_BUCKET_NAME = "test-bucket"

@pytest.fixture(scope="module")
def setup_test_directories():
    """Setup and teardown test directories."""
    os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEST_TEXT_INSTRUCTIONS_FOLDER, exist_ok=True)
    os.makedirs(TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER, exist_ok=True)
    os.makedirs(TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER, exist_ok=True)
    os.makedirs(TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER, exist_ok=True)
    yield
    shutil.rmtree(TEST_BASE_FOLDER)

# Unit Tests

from image_descriptions.cli import makedirs

@patch('image_descriptions.cli.image_descriptions_txt_folder', TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER)
@patch('image_descriptions.cli.image_descriptions_json_folder', TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER)
@patch('image_descriptions.cli.image_descriptions_jsonl_folder', TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER)
def test_makedirs():
    """Test that makedirs creates the required directories."""
    makedirs()
    assert os.path.exists(TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER)
    assert os.path.exists(TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER)
    assert os.path.exists(TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER)


@patch("image_descriptions.cli.storage.Client")
def test_upload_to_gcs(mock_storage_client):
    """Test the upload_to_gcs function."""
    # Ensure required directory exists
    makedirs()
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    test_file = os.path.join(TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER, "test_file.txt")
    os.makedirs(TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER, exist_ok=True)  # Ensure directory exists
    with open(test_file, "w") as f:
        f.write("mock content")

    upload_to_gcs(TEST_BUCKET_NAME ,test_file, "test_blob_name")
    mock_bucket.blob.assert_called_with("test_blob_name")
    mock_blob.upload_from_filename.assert_called_with(test_file)
    os.remove(test_file)

@patch("image_descriptions.cli.storage.Client")
def test_download_files_from_gcs(mock_storage_client):
    """Test downloading files from GCS."""
    # Ensure directories are created
    makedirs()

    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = MagicMock()
    mock_blob.name = "training/images/test_image.png"
    mock_bucket.list_blobs.return_value = [mock_blob]

    download_files_from_gcs(TEST_BUCKET_NAME, "training/images", TEST_IMAGES_FOLDER)
    downloaded_image = os.path.join(TEST_IMAGES_FOLDER, "test_image.png")
    
    # Simulate downloaded file creation
    os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)  # Ensure directory exists
    with open(downloaded_image, "w") as f:
        f.write("mock image content")
    
    assert os.path.exists(downloaded_image)
    os.remove(downloaded_image)


@patch("image_descriptions.cli.GenerativeModel")
@patch("image_descriptions.cli.Image.load_from_file")
def test_generate_image_description(mock_load_from_file, mock_generative_model):
    """Test image description generation."""
    mock_load_from_file.return_value = "mock_image"  # Mock image loading

    mock_model_instance = mock_generative_model.return_value
    mock_response = MagicMock()
    mock_response.text = "Mock description of the crochet image."
    mock_model_instance.generate_content.return_value = mock_response

    description = generate_image_description("mock_image_path")
    assert description == "Mock description of the crochet image."


@patch('image_descriptions.cli.image_descriptions_txt_folder', TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER)
def test_save_txt_file():
    """Test saving a description as a text file."""
    # Ensure the directory exists
    makedirs()

    test_description = "Mock description text."
    save_txt_file("test_image.png", test_description)
    txt_file_path = os.path.join(TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER, "test_image.txt")

    assert os.path.exists(txt_file_path), f"Expected file {txt_file_path} to exist."
    with open(txt_file_path, "r") as f:
        content = f.read()
    assert content == test_description
    os.remove(txt_file_path)


@patch('image_descriptions.cli.image_descriptions_json_folder', TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER)
def test_create_json_file():
    """Test creating a JSON file for image descriptions."""
    # Ensure the directory exists
    makedirs()

    test_user_text = "User text prompt."
    test_model_text = "Model generated text."
    create_json_file("test_image.png", test_user_text, test_model_text)

    json_file_path = os.path.join(TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER, "test_image.json")
    assert os.path.exists(json_file_path), f"Expected JSON file {json_file_path} to exist."

    with open(json_file_path, "r") as f:
        data = json.load(f)
    assert data["contents"][0]["role"] == "user"
    assert data["contents"][0]["parts"][0]["text"] == test_user_text
    assert data["contents"][1]["role"] == "model"
    assert data["contents"][1]["parts"][0]["text"] == test_model_text

    os.remove(json_file_path)

# Integration Tests

@patch('image_descriptions.cli.raw_image_folder', TEST_IMAGES_FOLDER)
@patch('image_descriptions.cli.image_descriptions_txt_folder', TEST_IMAGE_DESCRIPTIONS_TXT_FOLDER)
@patch('image_descriptions.cli.image_descriptions_json_folder', TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER)
@patch("image_descriptions.cli.generate_image_description", return_value="Generated description")
@patch("image_descriptions.cli.save_txt_file")
@patch("image_descriptions.cli.create_json_file")
def test_process(mock_create_json, mock_save_txt, mock_generate_description, setup_test_directories):
    """Test the end-to-end process of generating descriptions for images."""
    # Ensure directories are created
    makedirs()

    # Create a mock image file
    mock_image_path = os.path.join(TEST_IMAGES_FOLDER, "test_image.png")
    PILImage.new("RGB", (256, 256)).save(mock_image_path)

    process()

    # Check if save_txt_file and create_json_file were called
    mock_save_txt.assert_called_with("test_image.png", "Generated description")
    mock_create_json.assert_called_with("test_image.png", "Generated description", "No text instruction available.")

    os.remove(mock_image_path)



def test_split_json_to_jsonl(setup_test_directories):
    """Test splitting JSON files into train, validation, and test JSONL files."""
    # Create sample JSON files
    for i in range(5):
        json_data = {"contents": [{"role": "user", "parts": [{"text": f"User text {i}"}]},
                                  {"role": "model", "parts": [{"text": f"Model text {i}"}]}]}
        with open(os.path.join(TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER, f"test_{i}.json"), "w") as f:
            json.dump(json_data, f)
    
    split_json_to_jsonl(TEST_IMAGE_DESCRIPTIONS_JSON_FOLDER, TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER)

    assert os.path.exists(os.path.join(TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER, "train.jsonl"))
    assert os.path.exists(os.path.join(TEST_IMAGE_DESCRIPTIONS_JSONL_FOLDER, "test.jsonl"))

# System Test

@patch("image_descriptions.cli.download_files_from_gcs")
@patch("image_descriptions.cli.process")
@patch("image_descriptions.cli.split_json_to_jsonl")
@patch("image_descriptions.cli.upload")
def test_main_system_flow(mock_upload, mock_split, mock_process, mock_download):
    """System test to verify end-to-end functionality."""
    from image_descriptions.cli import main
    import argparse

    parser = argparse.ArgumentParser(description="Generate image descriptions and upload to GCS.")
    parser.add_argument("-d", "--download", action="store_true")
    parser.add_argument("-p", "--process", action="store_true")
    parser.add_argument("-s", "--split", action="store_true")
    parser.add_argument("-u", "--upload", action="store_true")
    parser.add_argument("-b", "--bucket", type=str, default="")

    # Run main function with all steps
    args = parser.parse_args(["--download", "--process", "--split", "--upload", "--bucket", TEST_BUCKET_NAME])
    main(args)

    # Check that each step was called
    assert mock_download.call_count == 2  # Called twice for images and text instructions
    mock_process.assert_called_once()
    mock_split.assert_called_once()
    mock_upload.assert_called_once()
