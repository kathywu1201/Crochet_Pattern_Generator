import numpy as np
from transformers import AutoFeatureExtractor, Swinv2Model
from PIL import Image
import torch
import io

def image_to_vector(image_path: str) -> np.ndarray:
    """
    Convert an image to a vector using a pre-trained model.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
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
    
    Args:
        image_bytes (bytes): The byte data of the image.
    
    Returns:
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
