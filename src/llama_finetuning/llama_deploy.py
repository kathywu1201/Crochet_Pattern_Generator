import modal
from fastapi import Request, UploadFile
import os
from PIL import Image
import requests

# Define Modal app
app = modal.App(name="llama-predict2")

# Configure secrets
huggingface_secret = modal.Secret.from_name("my-huggingface-secret")

# Paths for test images
test_images_path = "test/images"

# Create the Modal image with necessary dependencies
model_image = modal.Image.debian_slim().pip_install(
    "Pillow",
    "torch==2.4.0",
    "transformers==4.45.1",
    "huggingface_hub>=0.26.2",
    "peft",
    "accelerate>=0.26.0",
    "trl",
    "requests"
)

@app.cls(
    gpu="H100:2",
    timeout=8000,
    secrets=[huggingface_secret],
    image=model_image,
    mounts=[
        modal.Mount.from_local_dir(test_images_path, remote_path="/root/images"),
    ],
)
class LlamaModel:
    def __init__(self):
        self.processor = None
        self.model = None

    @modal.build()
    def download_model(self):
        """s
        Downloads model weights during the image build process.
        """
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        base_model = "ccwxp116/fine-tuned-visionllama-v3"
        AutoProcessor.from_pretrained(base_model).save_pretrained("/root/model")
        MllamaForConditionalGeneration.from_pretrained(base_model).save_pretrained("/root/model")

    @modal.enter()
    def load_model(self):
        """
        Loads the model and processor when the container starts.
        """
        from transformers import AutoProcessor, MllamaForConditionalGeneration
        import torch

        local_model_path = "/root/model"

        self.processor = AutoProcessor.from_pretrained(local_model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            local_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    @modal.web_endpoint(method="POST")
    async def predict(self, request: Request):
        try:
            # Parse form data
            form = await request.form()
            user_input = form.get("user_input", "")
            image_file = form.get("image", None)

            if image_file:
                # Save the uploaded image to a temporary file
                image_path = f"/tmp/{image_file.filename}"
                with open(image_path, "wb") as f:
                    f.write(await image_file.read())

                # Open the image
                img = Image.open(image_path)
            else:
                img = None

            # Dynamic prompt combining user preference and image
            structured_prompt = f"""
            Analyze the product shown in the image and provide a detailed crochet pattern.

            Respond in a clear and organized format, including the following sections:
            - **Crochet Product Name**
            - **Materials**
            - **Measurements**
            - **Abbreviations**
            - **Instructions**

            Adapt your response based on the user query:
            - If the user specifies a section (e.g., Materials), provide only that section.
            - If the user does not specify, provide the complete crochet pattern.

            User Query: {user_input}
            """

            # Process the input for the model
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": structured_prompt}
                ]}
            ]
            input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            if img:
                inputs = self.processor(img, input_text, return_tensors="pt").to(self.model.device)
            else:
                inputs = self.processor(text=input_text, return_tensors="pt").to(self.model.device)

            # Generate the response
            output = self.model.generate(**inputs, max_new_tokens=1000)

            # Decode the output
            decoded_output = self.processor.decode(output[0])
            # Extract content after the delimiter
            delimiter = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            if delimiter in decoded_output:
                relevant_output = decoded_output.split(delimiter, 1)[1].strip()
                if "Abbreviations:" in relevant_output:
                    relevant_output = relevant_output.split("Abbreviations:", 1)[0].strip()
            else:
                relevant_output = "Error: Delimiter not found in output."

            return {"output": relevant_output}

        except Exception as e:
            print(f"Error in predict endpoint: {e}")
            return {"error": str(e)}


@app.local_entrypoint()
def main():
    """
    Local entrypoint for testing.
    """
    llama = LlamaModel()
    llama.load_model()



