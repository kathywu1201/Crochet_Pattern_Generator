import modal
from pathlib import Path

app = modal.App(name="test_collate")

local_base_path = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset"
descriptions_path = f"{local_base_path}/descriptions"
images_path = f"{local_base_path}/images"
instructions_path = f"{local_base_path}/instructions"
cache_dir_local = f"{local_base_path}/cache"

import os
os.makedirs(cache_dir_local, exist_ok=True)

@app.function(
    gpu="A100",
    timeout=3600,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    mounts=[
        modal.Mount.from_local_dir(descriptions_path, remote_path="/root/dataset/descriptions"),
        modal.Mount.from_local_dir(images_path, remote_path="/root/dataset/images"),
        modal.Mount.from_local_dir(instructions_path, remote_path="/root/dataset/instructions"),
        modal.Mount.from_local_dir(cache_dir_local, remote_path="/root/.cache/huggingface"),  # Mount cache
    ],
    image=modal.Image.debian_slim().pip_install(
        "Pillow",  
        "datasets==3.0.1",
        "peft==0.13.0",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.45.1",
        "huggingface_hub>=0.26.2",
        "accelerate==0.34.2",
        "evaluate==0.4.3",
        "bitsandbytes==0.44.0",
        "trl==0.11.1",
        "qwen_vl_utils",
        "tensorboard"  
    )
)
def test_collate():
    import os
    from datasets import Dataset
    from PIL import Image
    import base64
    import io
    import torch  
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
    from huggingface_hub import login
    import torch  
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    import logging
    from pathlib import Path

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    # Mock Tokenizer Class
    class MockTokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, tokens):
            # Mock implementation: return a list of dummy token IDs
            return [1 for _ in tokens]

    # Mock Processor Class
    class MockProcessor:
        def __init__(self):
            self.tokenizer = MockTokenizer()
            self.image_token = "<|image|>"

        def apply_chat_template(self, messages, tokenize=False):
            """
            Mock method to simulate processor.apply_chat_template.
            Concatenates message contents.
            """
            return " ".join([msg["content"] for msg in messages])

        def __call__(self, text, images, return_tensors, padding):
            """
            Mock processor call method.
            Returns a dictionary mimicking the actual processor's output.
            """
            # Simulate tokenization by assigning dummy token IDs
            # For simplicity, assign a fixed sequence length
            sequence_length = 10
            input_ids = torch.randint(1, 100, (len(text), sequence_length))

            # Simulate image processing by creating random pixel values
            # Assuming images are resized to (3, 224, 224)
            pixel_values = torch.randn(len(images), 3, 224, 224)

            return {
                "input_ids": input_ids,
                "pixel_values": pixel_values
            }

    def process_vision_info(messages):
        """
        Mock function to extract image input from messages.
        Assumes the first message contains the image information.
        """
        content = messages[0]["content"]
        if "<|image|>data:image/png;base64," in content:
            # Extract the base64 string part
            base64_str = content.split("<|image|>data:image/png;base64,", 1)[1].split(" ", 1)[0]
            return [f"<|image|>data:image/png;base64,{base64_str}"]
        elif "<|image|>" in content:
            # Extract the image path part
            image_path = content.split("<|image|>", 1)[1].split(" ", 1)[0]
            return [f"<|image|>{image_path}"]
        else:
            return [None]

    def collate_fn(examples, processor):
        """
        Collate function to process batch examples.

        Args:
            examples (list): A list of examples from the dataset.
            processor (MockProcessor): The mock processor instance.

        Returns:
            dict: A batch dictionary with tokenized texts, processed images, and labels.
        """
        # Extract texts from messages
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]

        # Extract image inputs
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        # Process images
        loaded_images = []
        for idx, img_input in enumerate(image_inputs):
            if isinstance(img_input, str):
                if img_input.startswith("<|image|>data:image/"):
                    # It's a base64-encoded image
                    try:
                        # Extract the base64 string
                        base64_str = img_input.split("base64,", 1)[1].split(" ", 1)[0]
                        image_data = base64.b64decode(base64_str)
                        image = Image.open(io.BytesIO(image_data)).convert("RGB")
                        logging.info(f"Sample {idx}: Successfully decoded base64 image.")
                    except Exception as e:
                        logging.error(f"Sample {idx}: Error decoding base64 image: {e}")
                        # Create a blank image or handle as needed
                        image = Image.new("RGB", (224, 224))
                elif img_input.startswith("<|image|>"):
                    # It's an image path
                    image_path = img_input[len("<|image|>"):]
                    if os.path.exists(image_path):
                        try:
                            image = Image.open(image_path).convert("RGB")
                            logging.info(f"Sample {idx}: Successfully loaded image from path.")
                        except Exception as e:
                            logging.error(f"Sample {idx}: Error loading image from path {image_path}: {e}")
                            # Create a blank image or handle as needed
                            image = Image.new("RGB", (224, 224))
                    else:
                        logging.error(f"Sample {idx}: Image path does not exist: {image_path}")
                        # Create a blank image or handle as needed
                        image = Image.new("RGB", (224, 224))
                else:
                    logging.error(f"Sample {idx}: Unrecognized image input format: {img_input}")
                    # Create a blank image or handle as needed
                    image = Image.new("RGB", (224, 224))
            elif isinstance(img_input, Image.Image):
                # It's already a PIL Image
                image = img_input
                logging.info(f"Sample {idx}: Image is already a PIL Image.")
            elif img_input is None:
                logging.error(f"Sample {idx}: Image input is None.")
                # Create a blank image or handle as needed
                image = Image.new("RGB", (224, 224))
            else:
                logging.error(f"Sample {idx}: Invalid image input type: {type(img_input)}")
                # Create a blank image or handle as needed
                image = Image.new("RGB", (224, 224))
            
            loaded_images.append(image)

        # Tokenize texts and process images
        try:
            batch = processor(text=texts, images=loaded_images, return_tensors="pt", padding=True)
            logging.info("Batch processing successful.")
        except Exception as e:
            logging.error(f"Error during processor call: {e}")
            raise e

        # Process labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding tokens

        # Ignore image token indices in loss computation
        if hasattr(processor, 'image_token'):
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        else:
            image_tokens = []  # Adjust based on your actual processor's implementation

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch

    def test_collate_fn(collate_fn, processor):
        """
        Tests the collate_fn function with sample data using the mock processor.

        Args:
            collate_fn (function): The collate_fn to be tested.
            processor (MockProcessor): The mock processor instance.
        """
        # Sample base64-encoded PNG image (a tiny red dot)
        red_dot_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/"
            "Jc6DXQAAAABJRU5ErkJggg=="
        )
        base64_image = f"<|image|>data:image/png;base64,{red_dot_png}"

        # Path to an existing image (for testing, we'll create a temporary image)
        temp_image_path = "/tmp/test_image.png"
        Image.new("RGB", (224, 224), color=(0, 255, 0)).save(temp_image_path)  # Green image

        # Sample data representing various scenarios
        samples = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{base64_image} Describe this image in two sentences."
                    },
                    {
                        "role": "assistant",
                        "content": "This is a small red dot on a transparent background. It is centered in the image."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<|image|>{temp_image_path} Describe this image in two sentences."
                    },
                    {
                        "role": "assistant",
                        "content": "This is a solid green square image with dimensions 224x224 pixels. The green color is uniform across the entire image."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "<|image|>data:image/png;base64,INVALID_BASE64 Describe this image in two sentences."
                    },
                    {
                        "role": "assistant",
                        "content": "Unable to describe the image as it is corrupted or invalid."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "<|image|>/path/to/nonexistent_image.png Describe this image in two sentences."
                    },
                    {
                        "role": "assistant",
                        "content": "The image path provided does not exist. Unable to describe the image."
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "No image provided. Generate crochet instructions based on the following description."
                    },
                    {
                        "role": "assistant",
                        "content": "Generate crochet instructions without an image."
                    }
                ]
            }
        ]

        logging.info("===== Starting collate_fn Test =====")
        try:
            batch = collate_fn(samples, processor)
            logging.info("===== collate_fn Test Successful =====")
            logging.info("Batch Output:")
            logging.info(batch)
        except Exception as e:
            logging.error(f"An error occurred during collate_fn execution: {e}")
        finally:
            # Clean up the temporary image
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        logging.info("===== collate_fn Test Completed =====")
            # Instantiate the mock processor
    processor = MockProcessor()

            # Execute the test function
    test_collate_fn(collate_fn, processor)

if __name__ == "__main__":
    with app.run():
        test_collate()
