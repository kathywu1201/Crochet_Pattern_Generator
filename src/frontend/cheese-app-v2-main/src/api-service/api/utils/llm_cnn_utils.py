import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
import io
import json
import requests
import zipfile
import numpy as np
from PIL import Image
from pathlib import Path
from tempfile import TemporaryDirectory
import traceback
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from vertexai.generative_models import GenerativeModel, ChatSession, Part

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-1.5-flash-002"

# CNN Model details
AUTOTUNE = tf.data.experimental.AUTOTUNE
local_experiments_path = "/persistent/experiments"
best_model = None
best_model_id = None
cnn_model = None
data_details = None
image_width = 224
image_height = 224
num_channels = 3

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 3000,  # Maximum number of tokens for output
    "temperature": 0.1,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}
# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in cheese knowledge.

When answering a query:
1. Demonstrate expertise in cheese, including aspects like:
  - Production methods and techniques
  - Flavor profiles and characteristics
  - Aging processes and requirements
  - Regional varieties and traditions
  - Pairing recommendations
  - Storage and handling best practices
2. Always maintain a professional and knowledgeable tone, befitting a cheese expert.

Your goal is to provide accurate, helpful information about cheese for each query.
"""
generative_model = GenerativeModel(
	GENERATIVE_MODEL,
	system_instruction=[SYSTEM_INSTRUCTION]
)

# Initialize chat sessions
chat_sessions: Dict[str, ChatSession] = {}

def create_chat_session() -> ChatSession:
    """Create a new chat session with the model"""
    return generative_model.start_chat()

def generate_chat_response(chat_session: ChatSession, message: Dict) -> str:
    response = chat_session.send_message(
        message["content"],
        generation_config=generation_config
    )
    return response.text

def rebuild_chat_session(chat_history: List[Dict]) -> ChatSession:
    """Rebuild a chat session with complete context"""
    new_session = create_chat_session()
    
    for message in chat_history:
        if message["role"] == 'user' and message["content"] != "":
            prompt = message["content"]
            response = new_session.send_message(
                prompt,
                generation_config=generation_config
            )
        if message["role"] == 'cnn':
            prompt = f"We have already identified the image of a cheese as {message['results']['prediction_label']}"
            response = new_session.send_message(
                prompt,
                generation_config=generation_config
            )
    
    return new_session

def load_cnn_model():
    print("Loading CNN Model...")
    global cnn_model, data_details

    os.makedirs(local_experiments_path, exist_ok=True)

    best_model_path = os.path.join(
        local_experiments_path,
        "experiments",
        "mobilenetv2_train_base_True.keras"
    )
    if not os.path.exists(best_model_path):
        # Download from Github for easy access (This needs to be from you GCS bucket)
        # https://github.com/dlops-io/models/releases/download/v3.0/experiments.zip
        packet_url = "https://github.com/dlops-io/models/releases/download/v3.0/experiments.zip"
        packet_file = os.path.basename(packet_url)
        with requests.get(packet_url, stream=True, headers=None) as r:
            r.raise_for_status()
            with open(os.path.join(local_experiments_path, packet_file), "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with zipfile.ZipFile(os.path.join(local_experiments_path, packet_file)) as zfile:
            zfile.extractall(local_experiments_path)

    print("best_model_path:", best_model_path)
    cnn_model = tf.keras.models.load_model(best_model_path)
    print(cnn_model.summary())

    data_details_path = os.path.join(
        local_experiments_path, "experiments", "data_details.json"
    )

    # Load data details
    with open(data_details_path, "r") as json_file:
        data_details = json.load(json_file)

# Load the CNN Model
load_cnn_model()

def load_preprocess_image_from_path(image_path):
    print("Image", image_path)

    image_width = 224
    image_height = 224
    num_channels = 3

    # Prepare the data
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        return image

    # Normalize pixels
    def normalize(image):
        image = image / 255
        return image

    test_data = tf.data.Dataset.from_tensor_slices(([image_path]))
    test_data = test_data.map(load_image, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.repeat(1).batch(1)

    return test_data


def make_prediction(image_path):

    # Load & preprocess
    test_data = load_preprocess_image_from_path(image_path)

    # Make prediction
    prediction = cnn_model.predict(test_data)
    idx = prediction.argmax(axis=1)[0]
    prediction_label = data_details["index2label"][str(idx)]

    if cnn_model.layers[-1].activation.__name__ != "softmax":
        prediction = tf.nn.softmax(prediction).numpy()
        print(prediction)

    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction) * 100, 2)
    }