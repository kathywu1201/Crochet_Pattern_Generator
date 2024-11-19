import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock

# Mock the GCP dependencies before importing the module under test
# with patch('llm_rag.rag.TextEmbeddingModel') as MockTextEmbeddingModel:
#     mock_instance = MagicMock()
#     MockTextEmbeddingModel.from_pretrained.return_value = mock_instance
#     from llm_rag.rag import (
#         download,
#         generate_query_embedding,
#         generate_text_embeddings,
#         load_text_and_image_embeddings,
#         chunk,
#         embed,
#         load,
#         query,
#         re_rank_results,
#         upload
#     )

from llm_rag.rag import (
        download,
        generate_query_embedding,
        generate_text_embeddings,
        load_text_and_image_embeddings,
        chunk,
        embed,
        load,
        query,
        re_rank_results,
        upload
    )

# Setup and teardown functions for creating necessary directories and files
@pytest.fixture(scope="module")
def setup_folders():
    """Setup input and output folders before tests and teardown after."""
    os.makedirs("input_datasets", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    yield
    shutil.rmtree("input_datasets")
    shutil.rmtree("outputs")


@pytest.fixture
def sample_text_file(setup_folders):
    """Create a sample text file for testing."""
    file_path = "input_datasets/text_instructions/txt_outputs/sample.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write("Sample text for testing chunking.")
    return file_path


@pytest.fixture
def sample_image_embedding():
    """Create a sample image embedding as a .npy file for testing."""
    npy_path = "input_datasets/image_vectors/sample_book.npy"
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, np.random.rand(1024))  # Random 1024-dimensional embedding
    return npy_path

# Setup a mock collect to mock crombadb
class MockCollection:
    def __init__(self, name="mock_collection"):
        self.name = name
        self.data = []

    def add(self, ids, documents, metadatas, embeddings):
        # Append data to internal list for inspection, simulating storage
        self.data.append({
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings
        })


### Unit Tests ###
# def test_generate_query_embedding(): # correct
#     """Test embedding size and format."""
#     query = "sample query text"
#     embedding = generate_query_embedding(query)
#     assert isinstance(embedding, list), "Embedding should be a list"
#     assert len(embedding) == 256, "Embedding should be 256-dimensional"


# def test_generate_text_embeddings(): # correct
#     """Test batch generation of embeddings and size consistency."""
#     chunks = ["Text chunk 1", "Text chunk 2", "Text chunk 3"]
#     embeddings = generate_text_embeddings(chunks)
#     assert isinstance(embeddings, list), "Embeddings should be a list"
#     assert len(embeddings) == 3, "There should be one embedding per chunk"
#     assert all(len(embed) == 256 for embed in embeddings), "Each embedding should be 256-dimensional"


# def test_load_text_and_image_embeddings(sample_image_embedding): # Correct
#     """Test loading of text and image embeddings with combined size."""
#     data = {
#         "chunk": ["Sample text chunk"],
#         "book": ["sample_book"],
#         "embedding": [np.random.rand(256).tolist()],
#         "image_embedding": [np.random.rand(1024).tolist()]
#     }
#     df = pd.DataFrame(data)
#     mock_collection = MockCollection()
#     try:
#         load_text_and_image_embeddings(df, collection=mock_collection)
#         assert len(mock_collection.data) > 0, "Data was not added to the mock collection"
#     except Exception as e:
#         pytest.fail(f"load_text_and_image_embeddings raised an error: {e}")


### Integration Tests ###
# def test_chunk_creation(sample_text_file): # Correct
#     """Test the chunking of a sample text file."""
#     chunk()  # Assuming chunk uses global INPUT_FOLDER and OUTPUT_FOLDER
#     output_files = os.listdir("outputs")
#     assert any("chunks-sample" in f for f in output_files), "Output file not created for chunking"
#     assert len(output_files) > 0, "Expected chunk output files not created"


# def test_embedding_and_loading(sample_text_file, sample_image_embedding): # correct
#     """Test embedding generation and loading into a collection."""
#     embed()  # This should create embeddings in OUTPUT_FOLDER
#     output_files = os.listdir("outputs")
#     assert any("embeddings-sample" in f for f in output_files), "Embedding file not created"
#     assert len(output_files) > 0, "Expected embedding output files not created"


### System Tests ###
# @patch("llm_rag.rag.storage.Client")  # Mock the storage.Client class
# @patch("llm_rag.rag.chromadb.HttpClient")  # Mock ChromaDB HttpClient
# def test_full_pipeline(mock_chromadb_client, mock_storage_client, setup_folders, sample_text_file, sample_image_embedding):
#     """End-to-end test covering all stages with a mock bucket and mock ChromaDB."""

#     # Mock for storage.Client and bucket
#     mock_client_instance = MagicMock()
#     mock_bucket_instance = MagicMock()
#     mock_blob_instance = MagicMock()

#     # Configure return values for storage mocks
#     mock_storage_client.return_value = mock_client_instance
#     mock_client_instance.bucket.return_value = mock_bucket_instance
#     mock_bucket_instance.blob.return_value = mock_blob_instance

#     # Mock download and upload operations
#     mock_bucket_instance.list_blobs.return_value = []  # Simulate no files in bucket for download
#     mock_blob_instance.download_to_filename.side_effect = lambda filename: open(filename, "w").close()
#     mock_blob_instance.upload_from_filename.side_effect = lambda filename: None

#     # Mock ChromaDB client setup
#     mock_chromadb_instance = MagicMock()
#     mock_chromadb_client.return_value = mock_chromadb_instance

#     # Mock methods on the ChromaDB client as needed
#     mock_chromadb_instance.get_user_identity.return_value = MagicMock()
#     mock_collection = MagicMock()
#     mock_chromadb_instance.get_collection.return_value = mock_collection
#     mock_collection.add.return_value = None  # Simulate adding embeddings
#     mock_chromadb_instance.query.return_value = {"results": []}  # Simulate empty query results

#     # Run download (ensure it doesn't error on empty/bare setup)
#     download()

#     # Run chunk creation
#     chunk()
#     assert os.path.exists("outputs"), "Outputs directory should exist after chunking"

#     # Run embedding generation and loading
#     embed()
#     output_files = os.listdir("outputs")
#     assert any("embeddings" in f for f in output_files), "Embedding files not created as expected"

#     # Load into collection
#     load()

#     # Simulate a query to check pipeline up to ranking results
#     try:
#         query()
#     except Exception as e:
#         pytest.fail(f"Query function failed during end-to-end test: {e}")

#     # Check if upload runs without error
#     try:
#         upload()
#     except Exception as e:
#         pytest.fail(f"Upload function failed during end-to-end test: {e}")


### Test Ranking ###
def test_re_rank_results(): # Correct
    """Test re-ranking based on simulated text and image result dictionaries."""
    text_results = {"ids": [["doc1", "doc2", "doc3"]], "distances": [[0.2, 0.5, 0.8]]}
    image_results = {"ids": [["doc3", "doc1", "doc2"]], "distances": [[0.3, 0.4, 0.6]]}

    ranked_results = re_rank_results(text_results, image_results)
    assert isinstance(ranked_results, list), "Ranked results should be a list"
    assert ranked_results[0]["id"] == "doc1", "Doc1 should be ranked first based on weighting"
