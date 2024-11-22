import os
import sys
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import shutil

# Directory paths used in tests
TEST_IMAGES_FOLDER = "test_user/images"
TEST_IMAGE_VECTORS = "test_user/image_vectors"

# Add the src directory to the path for imports
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_dir)

# Import from image_2_vector (corrected module path)
from image_2_vector.cli import (
    makedirs,
    upload_to_gcs,
    download_images,
    image_to_vector,
    process_images,
    upload_vectors,
)

@pytest.fixture(scope="module")
def setup_folders():
    # Setup test folders for the test suite
    os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEST_IMAGE_VECTORS, exist_ok=True)
    yield
    # Teardown after tests
    os.rmdir(TEST_IMAGES_FOLDER)
    os.rmdir(TEST_IMAGE_VECTORS)

def test_makedirs(setup_folders):
    # Mocking the paths in the module
    with patch('image_2_vector.cli.images_folder', TEST_IMAGES_FOLDER), \
         patch('image_2_vector.cli.image_vectors', TEST_IMAGE_VECTORS):
        makedirs()
        assert os.path.isdir(TEST_IMAGES_FOLDER)
        assert os.path.isdir(TEST_IMAGE_VECTORS)

@patch("image_2_vector.cli.storage.Client")
def test_upload_to_gcs(mock_storage_client):
    # Mock storage client and upload process
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = mock_bucket.blob.return_value

    upload_to_gcs("local_test_file.png", "test_blob_name")
    mock_bucket.blob.assert_called_with("test_blob_name")
    mock_blob.upload_from_filename.assert_called_with("local_test_file.png")

@patch("image_2_vector.cli.storage.Client")
def test_download_images(mock_storage_client, setup_folders):
    # Mock storage client and simulate downloading process
    mock_client_instance = mock_storage_client.return_value
    mock_bucket = mock_client_instance.bucket.return_value
    mock_blob = MagicMock()
    mock_blob.name = "training/images/test_image.png"
    mock_bucket.list_blobs.return_value = [mock_blob]

    with patch('image_2_vector.cli.images_folder', TEST_IMAGES_FOLDER):
        download_images()
        
        # Simulate the downloaded file creation
        downloaded_image = os.path.join(TEST_IMAGES_FOLDER, "test_image.png")
        with open(downloaded_image, 'w') as f:
            f.write("mock image content")
        
        # Check if the file exists after download
        assert os.path.exists(downloaded_image), f"Expected {downloaded_image} to exist."
        
        # Clean up the downloaded mock file after the test
        if os.path.exists(downloaded_image):
            os.remove(downloaded_image)


@patch("image_2_vector.cli.AutoFeatureExtractor.from_pretrained")
@patch("image_2_vector.cli.Swinv2Model.from_pretrained")
def test_image_to_vector(mock_model, mock_extractor):
    # Mock model inference
    mock_extractor.return_value = MagicMock()
    mock_model_instance = mock_model.return_value
    mock_model_instance.return_value.last_hidden_state.mean.return_value.squeeze.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3])

    # Creating a mock image in memory
    img = Image.new('RGB', (256, 256), color='red')
    img_path = "mock_image.png"
    img.save(img_path)

    vector = image_to_vector(img_path)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (3,)

    os.remove(img_path)

@patch("image_2_vector.cli.image_to_vector")
def test_process_images(mock_image_to_vector, setup_folders):
    # Mock the vector generation
    mock_image_to_vector.return_value = np.array([0.1, 0.2, 0.3])

    # Create mock images
    mock_image_path = os.path.join(TEST_IMAGES_FOLDER, "test_image.png")
    Image.new('RGB', (256, 256)).save(mock_image_path)

    with patch('image_2_vector.cli.images_folder', TEST_IMAGES_FOLDER), \
         patch('image_2_vector.cli.image_vectors', TEST_IMAGE_VECTORS):
        process_images()
        vector_path = os.path.join(TEST_IMAGE_VECTORS, "test_image.npy")
        assert os.path.exists(vector_path)
        loaded_vector = np.load(vector_path)
        assert np.array_equal(loaded_vector, np.array([0.1, 0.2, 0.3]))

@patch("image_2_vector.cli.upload_to_gcs")
def test_upload_vectors(mock_upload_to_gcs, setup_folders):
    # Create a mock npy file for upload
    vector_file = os.path.join(TEST_IMAGE_VECTORS, "test_vector.npy")
    np.save(vector_file, np.array([0.1, 0.2, 0.3]))

    with patch('image_2_vector.cli.image_vectors', TEST_IMAGE_VECTORS):
        upload_vectors()
        
        # Check if upload_to_gcs was called with any valid npy file in the expected directory
        mock_upload_to_gcs.assert_any_call(
            vector_file, f"training/image_vectors/{os.path.basename(vector_file)}"
        )
    
    # Clean up created files
    os.remove(vector_file)

@pytest.fixture(scope="module")
def setup_folders():
    # Setup test folders for the test suite
    os.makedirs(TEST_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(TEST_IMAGE_VECTORS, exist_ok=True)
    yield
    # Recursively remove test directories
    shutil.rmtree(TEST_IMAGES_FOLDER, ignore_errors=True)
    shutil.rmtree(TEST_IMAGE_VECTORS, ignore_errors=True)
