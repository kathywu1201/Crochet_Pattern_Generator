<<<<<<< HEAD
from transformers import AutoImageProcessor, TFAutoModel
from PIL import Image
import tensorflow as tf
import io
import numpy as np

def image_to_vector(image_path: str) -> np.ndarray:
    """
    Convert an image to a vector using a pre-trained Vision Transformer model with TensorFlow.
=======
import numpy as np
from transformers import AutoFeatureExtractor, Swinv2Model
from PIL import Image
import torch
import io

def image_to_vector(image_path: str) -> np.ndarray:
    """
    Convert an image to a vector using a pre-trained model.
>>>>>>> main
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
<<<<<<< HEAD
        np.ndarray: A 1024-dimensional image vector.
    """
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

    # Add a dense layer to ensure output vector size is 1024
    dense_layer = tf.keras.layers.Dense(1024, activation=None)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    
    # Use dense layer to reshape the vector
    pooled_output = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    reshaped_output = dense_layer(pooled_output)

    return reshaped_output.numpy().squeeze()

def image_to_vector_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert an image from bytes to a vector using a pre-trained Vision Transformer model with TensorFlow.
=======
        np.ndarray: The image vector.
    """
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    model = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    image = Image.open(image_path).convert("RGB")
    inputs = extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def image_to_vector_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert an image from bytes to a vector using a pre-trained model.
>>>>>>> main
    
    Args:
        image_bytes (bytes): The byte data of the image.
    
    Returns:
<<<<<<< HEAD
        np.ndarray: A 1024-dimensional image vector.
    """
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

    # Add a dense layer to ensure output vector size is 1024
    dense_layer = tf.keras.layers.Dense(1024, activation=None)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="tf")
    outputs = model(**inputs)

    # Use dense layer to reshape the vector
    pooled_output = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    reshaped_output = dense_layer(pooled_output)

    return reshaped_output.numpy().squeeze()
=======
        np.ndarray: The image vector.
    """
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    model = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    
    # Create a PIL image from the byte data
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
>>>>>>> main
