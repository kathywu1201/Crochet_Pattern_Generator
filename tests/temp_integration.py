from fastapi import FastAPI
from pydantic import BaseModel
import base64

app = FastAPI()


class OperationInput(BaseModel):
    file_path: str
    prompt: str

@app.post("/vector_generator")
async def vector_generator(input_date: OperationInput):
    prompt = input_data.prompt
    image_path = input_data.file_path
    vector = image_to_vector(image_path)
    vector_list = vector.tolist()

    # Encode the image to Base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Use appropriate format
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {"vector": vector_list, "image": f"{image_path}", "prompt": prompt} # image_path?



class OperationInput(BaseModel):
    vector: list()
    image: str
    prompt: str
    
app.post("/rag")
async def rag_query(input_data: OperationInput):
    vector = input_data.vector
    image_path = input_data.image
    prompt = input_data.prompt

    output = query(prompt, image_path, image_vector)
    return {"prompt": output, image: f"{image_path}"}

import requests

def test_vector_to_rag():
    vector_response = requests.post("http://localhost:5000/add", json={"file_path": ""})
    vector_result = vector_response.josn()["vector"]
    assert vector

    rag_response = requests.post(
        "http://localhost:5000/add", 
        json={
            "vector": vector_result["vector"],
            "image": vector_result["image"],
            "prompt": vector_result["prompt"]})
    rag_result = rag_response.josn()["prompt"]