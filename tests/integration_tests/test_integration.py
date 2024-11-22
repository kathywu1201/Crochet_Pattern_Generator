import requests
import numpy as np
import os
from unittest.mock import patch
# from app import mock_image_to_vector, mock_query

# # Integration test
# @patch("app.mock_image_to_vector", side_effect=mock_image_to_vector)
# @patch("app.mock_query", side_effect=mock_query)
def test_integration_vector_rag():
    # Start FastAPI server in a subprocess
    # process = subprocess.Popen(["uvicorn", "tests.app:app", "--host", "127.0.0.1", "--port", "8000"])
    # time.sleep(5)  # Give the server some time to start

    # Step 1: Test /vector_generator
    vector_response = requests.post(
        "http://image_2_vector:5000/vector_generator",
        json={"image_path": "mock_image_path.png"}
    )
    assert vector_response.status_code == 200, "Failed to get vector"

    # Save the downloaded vector file locally
    with open("test_vector.npy", "wb") as f:
        f.write(vector_response.content)

    # Load the vector to verify it is a NumPy array
    vector = np.load("test_vector.npy")
    assert isinstance(vector, np.ndarray), "Vector is not a NumPy array"
    assert vector.shape == (1024,), f"Vector shape is incorrect, got {vector.shape}"

    # Step 2: Test /rag
    rag_response = requests.post(
        "http://llm_rag:5001/rag",
        json={
            "user_query": "Describe the crochet pattern in the following image.",
            "image_vector": "test_vector.npy"
        }
    )
    # Print detailed error information
    if rag_response.status_code != 200:
        print(f"RAG Response Status Code: {rag_response.status_code}")
        print(f"RAG Response Content: {rag_response.text}")
        
    assert rag_response.status_code == 200, f"Failed to get RAG query response: {rag_response.text}"
    rag_result = rag_response.json()

    # Assert output is user_query + chunked_text
    expected_output = {"prompt": "Describe the crochet pattern in the following image. Mock chunked text generated by RAG."}
    assert rag_result == expected_output, f"Unexpected output: {rag_result}"
    print(f"Integration Test Passed. Output: {rag_result}")

    # Clean up temporary files
    os.remove("test_vector.npy")