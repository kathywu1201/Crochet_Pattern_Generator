import modal
import os
import torch

app = modal.App(name="llama-finetuning")

local_base_path = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset"
descriptions_path = f"{local_base_path}/descriptions"
images_path = f"{local_base_path}/images"
instructions_path = f"{local_base_path}/instructions"

@app.function(gpu="A10G", timeout=3600,
              secrets=[modal.Secret.from_name("my-huggingface-secret")],
              mounts=[
                  modal.Mount.from_local_dir(descriptions_path, remote_path="/root/dataset/descriptions"),
                  modal.Mount.from_local_dir(images_path, remote_path="/root/dataset/images"),
                  modal.Mount.from_local_dir(instructions_path, remote_path="/root/dataset/instructions"),
                  ],
              image=modal.Image.debian_slim().pip_install(
                  "Pillow",  
                  "datasets",
                  "peft",
                  "torch",
                  "transformers>=4.30.0",
                  "huggingface_hub>=0.26.2"
              ))
def train_llama():
    from datasets import Dataset
    from PIL import Image
    import base64
    import io
    from transformers import AutoProcessor, Trainer, TrainingArguments
    from huggingface_hub import login

    # Mock model definition for testing
    class MockModel:
        def __init__(self, *args, **kwargs):
            print("Initialized Mock Model")

        def forward(self, *args, **kwargs):
            print("Mock forward pass called")
            return {"loss": torch.tensor(0.0), "logits": torch.zeros((1, 10))}

        def save_pretrained(self, path):
            print(f"Mock model saved to {path}")

        def to(self, device):
            print(f"Mock model moved to {device}")
            return self

        def train(self):
            print("Mock model set to training mode")
            return self

        def eval(self):
            print("Mock model set to evaluation mode")
            return self

    # Determine whether to use the mock model for testing
    use_mock_model = os.getenv("USE_MOCK_MODEL", "false").lower() == "true"

    # Hugging Face login (can be bypassed if not needed for testing)
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Skipping Hugging Face login for testing.")

    # Paths for images and data
    image_folder = "/root/dataset/images"
    image_description_folder = "/root/dataset/descriptions"
    instruction_folder = "/root/dataset/instructions"

    # Helper function to read files and map by filename
    def load_data_from_folder(folder_path):
        data = {}
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            file_key = os.path.splitext(filename)[0]
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    data[file_key] = file.read().strip()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                    data[file_key] = file.read().strip()
        return data

    # Load data from each folder
    image_descriptions = load_data_from_folder(image_description_folder)
    instructions = load_data_from_folder(instruction_folder)

    # Load images and convert to base64 format for inclusion in the dataset
    def load_images_as_base64(image_folder):
        images = {}
        for filename in os.listdir(image_folder):
            filepath = os.path.join(image_folder, filename)
            file_key = os.path.splitext(filename)[0]
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                with open(filepath, 'rb') as img_file:
                    image = Image.open(img_file).convert("RGB")
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    images[file_key] = img_base64
            else:
                print(f"Skipping non-image file: {filename}")
        return images

    images = load_images_as_base64(image_folder)

    # Create a dataset by combining the data based on filename
    combined_data = []
    for key in image_descriptions.keys():
        if key in images and key in instructions:
            sample = {
                "Image Description": image_descriptions[key],
                "image": images[key],
                "Instruction": instructions[key],
            }
            combined_data.append(sample)

    # Mock dataset preparation
    dataset = [sample for sample in combined_data]  # Use combined data directly

    # Use either the real or mock model
    if use_mock_model:
        model = MockModel()
    else:
        from transformers import AutoModelForCausalLM
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).cuda()

    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir="fine-tuned-visionllama",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
    )

    # Data Collator
    def collate_fn(examples):
        texts = [processor(text=example["Image Description"], images=None, return_tensors="pt", padding=True) for example in examples]
        return texts

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    # Train the model (or mock)
    trainer.train()

    # Save the model
    model.save_pretrained("/path/to/save/mock_or_real_model")

if __name__ == "__main__":
    with app.run():
        train_llama()
