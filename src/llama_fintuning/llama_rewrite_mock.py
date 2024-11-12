import os
from datasets import Dataset
from PIL import Image
import modal

# Modal setup
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, PeftModel
from PIL import Image
import modal

# Modal setup
app = modal.App(name="llama-finetuning")

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
        modal.Mount.from_local_dir(instructions_path, remote_path="/root/dataset/instructions")
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
def test_batch_processing():
    from datasets import Dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    import torch
    from torch.utils.data import DataLoader

    # Mock data loading function
    def load_data():
        data = []
        for filename in os.listdir("/root/dataset/descriptions"):
            if filename.startswith("."):
                continue
            base_name = os.path.splitext(filename)[0]
            description_path = os.path.join("/root/dataset/descriptions", filename)
            image_path = os.path.join("/root/dataset/images", f"{base_name}.png")
            instruction_path = os.path.join("/root/dataset/instructions", f"{base_name}.txt")

            if os.path.exists(description_path) and os.path.exists(image_path) and os.path.exists(instruction_path):
                with open(description_path, 'r') as f:
                    description = f.read().strip()  # Added .strip() to remove trailing newlines
                with open(instruction_path, 'r') as f:
                    instruction = f.read().strip()
                data.append({"description": description, "image_path": image_path, "instruction": instruction})
        return Dataset.from_list(data)


    # Load dataset
    dataset = load_data()
    for sample in dataset:  # Replace `dataset` with your dataset object
        if not isinstance(sample["description"], (str, list)):
            print("Invalid data found:", sample)
            # Optionally, handle the invalid data by skipping, fixing, or raising an error
            continue  # Or handle based on your specific needs


    # Tokenizer (mock tokenizer for demonstration purposes)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Mock preprocessing function
    def preprocess_function(samples):
        images = []
        inputs = []
        labels = []
        
        # Iterate over each sample in the batch
        for image_path, description, instruction in zip(
            samples["image_path"], samples["description"], samples["instruction"]
        ):
            image = Image.open(image_path).convert("RGB")
            processed_inputs = processor(
                text=description,
                images=image,
                return_tensors="pt",
                padding=True
            )
            images.append(processed_inputs["pixel_values"])
            inputs.append(processor.tokenizer(description, return_tensors="pt").input_ids)
            labels.append(processor.tokenizer(instruction, return_tensors="pt").input_ids)

        return {
            "pixel_values": torch.cat(images),
            "input_ids": torch.cat(inputs),
            "labels": torch.cat(labels)
        }


    # Preprocess data
    dataset = dataset.map(preprocess_function, batched=True)

    # Preprocess dataset
    dataset = dataset.map(preprocess_function, batched=False)

    # Data collator and DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)

    # Check batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch}")
        if not batch or (batch["input_ids"].size(0) == 0):
            print(f"Batch {batch_idx} is empty!")
        else:
            print(f"Batch {batch_idx} processed successfully with size {batch['input_ids'].size()}")


# Entry point for Modal execution
if __name__ == "__main__":
    with app.run():
        test_batch_processing()