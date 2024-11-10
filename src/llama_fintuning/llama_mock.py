import modal
import os
import torch
from transformers import PreTrainedModel, PretrainedConfig

app = modal.App(name="llama-finetuning")
os.environ["USE_MOCK_MODEL"] = "true"  # Manually set the environment variable

# Define paths
local_base_path = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset"
descriptions_path = f"{local_base_path}/descriptions"
images_path = f"{local_base_path}/images"
instructions_path = f"{local_base_path}/instructions"

@app.function(
    gpu="A10G", 
    timeout=3600,
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
    )
)
def train_llama():
    from datasets import Dataset
    from PIL import Image
    import base64
    import io
    from transformers import AutoProcessor, Trainer, TrainingArguments
    from huggingface_hub import login

    # Define a mock model for testing purposes
    class MockConfig(PretrainedConfig):
        model_type = "mock_model"

    class MockModel(PreTrainedModel):
        config_class = MockConfig

        def __init__(self):
            config = MockConfig()
            super().__init__(config)
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

    # Use mock model if the environment variable is set
    use_mock_model = os.getenv("USE_MOCK_MODEL", "false").lower() == "true"

    # Hugging Face login
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Skipping Hugging Face login for testing.")

    # Define paths for data
    image_folder = "/root/dataset/images"
    image_description_folder = "/root/dataset/descriptions"
    instruction_folder = "/root/dataset/instructions"

    # Load data from folder
    def load_data_from_folder(folder_path):
        data = {}
        for filename in os.listdir(folder_path):
            if filename.startswith('.'):  # Skip hidden files
                continue
            filepath = os.path.join(folder_path, filename)
            file_key = os.path.splitext(filename)[0]  # Remove file extension to use as key
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    data[file_key] = file.read().strip()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                    data[file_key] = file.read().strip()
        return data

    # Load data
    image_descriptions = load_data_from_folder(image_description_folder)
    instructions = load_data_from_folder(instruction_folder)

    # Load images and convert to base64
    def load_images_as_base64(image_folder):
        images = {}
        for filename in os.listdir(image_folder):
            filepath = os.path.join(image_folder, filename)
            file_key = os.path.splitext(filename)[0]
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                try:
                    with open(filepath, 'rb') as img_file:
                        image = Image.open(img_file).convert("RGB")
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        images[file_key] = img_base64
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
            else:
                print(f"Skipping non-image file: {filename}")
        print(f"Loaded {len(images)} images.")
        return images

    images = load_images_as_base64(image_folder)

    # Create combined dataset
    combined_data = []
    for key in image_descriptions.keys():
        if key in images and key in instructions:
            if image_descriptions[key] and images[key] and instructions[key]:
                sample = {
                    "Image_Description": image_descriptions[key],
                    "image": images[key],
                    "Instruction": instructions[key],
                }
                combined_data.append(sample)
            else:
                print(f"Empty data for key: {key}")
        else:
            missing_keys = []
            if key not in images:
                missing_keys.append("images")
            if key not in instructions:
                missing_keys.append("instructions")
            print(f"Missing data for key: {key} in {', '.join(missing_keys)}")

    if not combined_data:
        print("No valid data found.")
    else:
        print(f"Number of valid examples: {len(combined_data)}")

    # Mock dataset preparation
    dataset = [sample for sample in combined_data]

    # Conditional model loading
    if use_mock_model:
        print("Using Mock Model")
        model = MockModel()
    else:
        print("Using Real Model")
        from transformers import AutoModelForCausalLM
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).cuda()

    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

    # Define Training Arguments
    gradient_checkpointing = not use_mock_model

    training_args = TrainingArguments(
        output_dir="fine-tuned-visionllama",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        save_steps=10,
        save_total_limit=2,
        logging_dir="./logs",
    )

    # Data Collator
    def collate_fn(examples):
        if not examples:
            print("No examples received in collate_fn.")
            return {}

        valid_examples = [example for example in examples if "Image_Description" in example and example["Image_Description"]]

        if not valid_examples:
            print("No valid examples found in collate_fn.")
            return {}

        inputs = []
        for example in valid_examples:
            processed = processor(text=example["Image_Description"], images=None, return_tensors="pt", padding=True)
            inputs.append(processed)

        if inputs:
            try:
                batch = {k: torch.cat([input[k] for input in inputs], dim=0) for k in inputs[0]}
                print("Batch created in collate_fn:", batch)
                return batch
            except Exception as e:
                print(f"Error creating batch: {e}")
                return {}
        else:
            print("No inputs to create batch in collate_fn.")
            return {}


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
