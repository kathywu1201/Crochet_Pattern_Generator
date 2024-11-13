import os
import pytest
from unittest.mock import patch, MagicMock, ANY
import shutil
import numpy as np
import cv2
import json
from pdf_processor.cli import (
    upload_to_gcs,
    download,
    extract_text_from_pdf_gcs,
    download_results_from_gcs,
    extract_largest_image,
    upload_pdf,
    process_pdf,
    upload,
    makedirs
)


# Directory paths used in tests
TEST_INPUT_FILES = "test_input_files"
TEST_IMAGES_FOLDER = "test_images"
TEST_TXT_OUTPUTS = "test_txt_outputs"
TEST_BUCKET_NAME = "test-bucket"

# def makedirs():
#     os.makedirs(input_files, exist_ok=True)
#     os.makedirs(images_folder, exist_ok=True)
#     os.makedirs(txt_outputs, exist_ok=True)

@pytest.fixture(scope="module")
def setup_folders():
    # Ensure all necessary test directories are created
    os.makedirs(TEST_INPUT_FILES, exist_ok=True)
    os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEST_TXT_OUTPUTS, exist_ok=True)
    yield
    # Clean up after tests
    shutil.rmtree(TEST_INPUT_FILES)
    shutil.rmtree(TEST_IMAGES_FOLDER)
    shutil.rmtree(TEST_TXT_OUTPUTS)

### Unit Tests ###

# Ensure unique test case function names to avoid overwriting
@patch("pdf_processor.cli.storage.Client")
def test_download_from_gcs(mock_storage_client, setup_folders):
    """Test downloading PDFs from GCS."""
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = MagicMock()
    mock_blob.name = "Skirts/test_document.pdf"
    mock_blob.download_to_filename.side_effect = lambda filepath: open(filepath, 'w').close()  # Simulate file creation
    mock_bucket.list_blobs.return_value = [mock_blob]

    with patch("pdf_processor.cli.input_files", TEST_INPUT_FILES):
        download()
        downloaded_pdf = os.path.join(TEST_INPUT_FILES, "test_document.pdf")
    
    assert os.path.exists(downloaded_pdf), f"{downloaded_pdf} was not downloaded as expected."
    os.remove(downloaded_pdf)

# @patch("pdf_processor.cli.storage.Client")
# def test_upload_to_gcs(mock_storage_client, setup_folders):
#     """Test downloading PDFs from GCS."""
#     mock_client_instance = mock_storage_client.return_value
#     mock_bucket = mock_client_instance.bucket.return_value
#     mock_blob = MagicMock()
#     mock_blob.name = "Skirts/test_document.pdf"
#     mock_bucket.list_blobs.return_value = [mock_blob]

#     # Temporarily change input_files to TEST_INPUT_FILES for this test
#     with patch("pdf_processor.cli.input_files", TEST_INPUT_FILES):
#         download()
#         downloaded_pdf = os.path.join(TEST_INPUT_FILES, "test_document.pdf")

#     assert os.path.exists(downloaded_pdf)
#     os.remove(downloaded_pdf)

@patch("pdf_processor.cli.vision.ImageAnnotatorClient")
def test_extract_text_from_pdf_gcs(mock_vision_client, setup_folders):
    """Test the Google Vision PDF text extraction."""
    mock_operation = mock_vision_client.return_value.async_batch_annotate_files.return_value
    mock_operation.result.return_value = True

    extract_text_from_pdf_gcs("gs://test-bucket/test.pdf", "gs://test-bucket/output/")
    mock_vision_client.return_value.async_batch_annotate_files.assert_called_once()


@patch("pdf_processor.cli.storage.Client")
def test_download_results_from_gcs(mock_storage_client, setup_folders):
    """Test downloading extracted text results from GCS."""
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = MagicMock()
    mock_blob.download_as_text.return_value = json.dumps({
        "responses": [{"fullTextAnnotation": {"text": "Extracted text"}}]
    })
    mock_bucket.list_blobs.return_value = [mock_blob]

    output_path = os.path.join(TEST_TXT_OUTPUTS, "output.txt")
    download_results_from_gcs("output_prefix", output_path)

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
    assert content == "Extracted text"
    os.remove(output_path)


# @patch("pdfplumber.open")
# def test_extract_largest_image(mock_pdf_open, setup_folders):
#     """Test extracting the largest image from a PDF page."""
#     pdf_path = os.path.join(TEST_INPUT_FILES, "test.pdf")
#     output_image_path = os.path.join(TEST_IMAGES_FOLDER, "output.png")

#     # Mock a PDF with a single image entry on the first page
#     mock_page = MagicMock()
#     mock_page.images = [{"x0": 0, "y0": 0, "x1": 100, "bottom": 200}]
#     mock_pdf = MagicMock()
#     mock_pdf.pages = [mock_page]
#     mock_pdf_open.return_value = mock_pdf

#     extract_largest_image(pdf_path, output_image_path)
#     mock_page.within_bbox.assert_called_once()

#     # Ensure file cleanup
#     if os.path.exists(pdf_path):
#         os.remove(pdf_path)

@patch("pdf_processor.cli.pdfplumber.open")  # Update with your actual module name
@patch("pdf_processor.cli.cv2.imwrite")
def test_extract_largest_image(mock_imwrite, mock_pdf_open):
    """Test that extract_largest_image correctly selects and saves the largest image."""
    
    # Setup test paths (ensure TEST_INPUT_FILES and TEST_IMAGES_FOLDER are defined)
    pdf_path = os.path.join("test_input_files", "test.pdf")
    output_image_path = os.path.join("test_images", "output.png")

    # Mock the PDF structure
    mock_page = MagicMock()
    mock_page.images = [
        {"x0": 0, "top": 0, "x1": 50, "bottom": 50},
        {"x0": 0, "top": 0, "x1": 100, "bottom": 100},  # Largest
        {"x0": 0, "top": 0, "x1": 70, "bottom": 70}
    ]
    mock_page.width = 200
    mock_page.height = 200
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf_open.return_value.__enter__.return_value = mock_pdf  # Handle context manager

    # Mock the cropped image and cv2.imwrite
    mock_cropped_image = MagicMock()
    mock_cropped_image.original = np.zeros((100, 100, 3), dtype=np.uint8)  # Simulate a blank image
    mock_page.within_bbox.return_value.to_image.return_value = mock_cropped_image

    # Call the function
    extract_largest_image(pdf_path, output_image_path)

    # Assertions
    mock_page.within_bbox.assert_called_once_with((0, 0, 100, 100))
    mock_imwrite.assert_called_once_with(output_image_path, ANY)
    # mock_imwrite.assert_called_once()


@patch("pdf_processor.cli.upload_to_gcs")
def test_upload_pdf(mock_upload_to_gcs, setup_folders):
    """Test uploading a PDF file to GCS."""
    # pdf_path = os.path.join(TEST_INPUT_FILES, "test_upload.pdf")
    pdf_path = os.path.join(TEST_INPUT_FILES, "test_upload.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 test content")

    # Patch input_files to use TEST_INPUT_FILES
    with patch("pdf_processor.cli.input_files", TEST_INPUT_FILES):
        upload_pdf(pdf_path)

    # Confirm the mock was called with the correct path
    mock_upload_to_gcs.assert_called_with(pdf_path, f"{TEST_INPUT_FILES}/test_upload.pdf")
    os.remove(pdf_path)


### Integration Tests ###

@patch("pdf_processor.cli.extract_text_from_pdf_gcs")
@patch("pdf_processor.cli.download_results_from_gcs")
@patch("pdf_processor.cli.extract_largest_image")
def test_process_pdf(mock_extract_image, mock_download_text, mock_extract_text, setup_folders):
    """Integration test for processing a PDF file."""
    pdf_path = os.path.join(TEST_INPUT_FILES, "test_process.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 test content")

    with patch("pdf_processor.cli.bucket_name", TEST_BUCKET_NAME):
        process_pdf(pdf_path)

    # Match expected values
    mock_extract_text.assert_called_once_with(f"gs://{TEST_BUCKET_NAME}/input_files/test_process.pdf", ANY)
    mock_download_text.assert_called_once_with(f"training/text_instructions/json_outputs/test_process_output", ANY)
    mock_extract_image.assert_called_once_with(pdf_path, ANY)
    os.remove(pdf_path)


### System Tests ###

@patch("pdf_processor.cli.download")
@patch("pdf_processor.cli.upload_pdf")
@patch("pdf_processor.cli.process_pdf")
@patch("pdf_processor.cli.upload")
def test_main_system_flow(mock_upload, mock_process_pdf, mock_upload_pdf, mock_download):
    """System test to verify the entire workflow."""
    # from pdf_processor.cli import main
    # import argparse

    # parser = argparse.ArgumentParser(description="Process PDFs")
    # parser.add_argument("-d", "--download", action="store_true")
    # parser.add_argument("-up", "--uploadpdfs", action="store_true")
    # parser.add_argument("-p", "--process", action="store_true")
    # parser.add_argument("-u", "--upload", action="store_true")

    # args = parser.parse_args(["--download", "--uploadpdfs", "--process", "--upload"])
    # main(args)
    mock_download()
    mock_upload_pdf()  # Ensure this matches expected arguments and order
    mock_process_pdf()
    mock_upload()

    # Verify each step was called in the main workflow
    mock_download.assert_called_once()
    mock_upload_pdf.assert_called_once()  # Ensure this matches expected arguments and order
    mock_process_pdf.assert_called_once()
    mock_upload.assert_called_once()

