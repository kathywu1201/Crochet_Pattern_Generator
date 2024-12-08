# Used for integration test
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import numpy as np

# Mock function for image_to_vector
def mock_image_to_vector(image_path):
    # Return a 1024-dimensional numpy vector
    return np.random.rand(1024)

app = FastAPI()

# Input model for the /vector_generator endpoint
class VectorInput(BaseModel):
    image_path: str

@app.post("/vector_generator")
async def vector_generator(input_data: VectorInput):
    vector = mock_image_to_vector(input_data.image_path)  # Replace with real function in production
    vector_file_path = "vector.npy"
    np.save(vector_file_path, vector)
    return FileResponse(vector_file_path, media_type="application/octet-stream")