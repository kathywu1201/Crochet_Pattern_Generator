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
    gpu="A100",
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
def train_llama():
    from datasets import Dataset
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
    from peft import PeftModel, LoraConfig
    import os
    from huggingface_hub import login

    # Hugging Face login
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")

    # Paths for images and data
    image_folder = "/root/dataset/images"
    descriptions_folder = "/root/dataset/descriptions"
    instructions_folder = "/root/dataset/instructions"

    # Load data
    def load_data():
        data = []
        for filename in os.listdir(descriptions_folder):
            base_name = os.path.splitext(filename)[0]
            description_path = os.path.join(descriptions_folder, filename)
            image_path = os.path.join(image_folder, f"{base_name}.png")  # Assuming images are .jpg
            instruction_path = os.path.join(instructions_folder, f"{base_name}.txt")

            if os.path.exists(description_path) and os.path.exists(image_path) and os.path.exists(instruction_path):
                with open(description_path, 'r') as f:
                    description = f.read()
                with open(instruction_path, 'r') as f:
                    instruction = f.read()
                data.append({"description": description, "image_path": image_path, "instruction": instruction})
        return Dataset.from_list(data)

    dataset = load_data()

    # Load model, tokenizer, and processor
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    # Preprocess function
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


    # Setup LoRA configuration for fine-tuning
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    peft_model = PeftModel(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned-llama",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        gradient_accumulation_steps=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        logging_dir='./logs',
        logging_steps=10,
        push_to_hub=False
    )

    # Trainer setup
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    peft_model.save_pretrained("./fine-tuned-llama")
    tokenizer.save_pretrained("./fine-tuned-llama")

# Entry point for Modal execution
if __name__ == "__main__":
    with app.run():
        train_llama()
