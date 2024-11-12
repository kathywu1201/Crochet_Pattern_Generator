import modal

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
    import os
    from datasets import Dataset
    from PIL import Image
    import base64
    import io
    import torch  
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
    from huggingface_hub import login

    # Hugging Face login
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")


    # Paths for images and data
    image_folder = "/root/dataset/images"
    image_description_folder = "/root/dataset/descriptions"
    instruction_folder = "/root/dataset/instructions"

    # Helper function to read files and map by filename
    def load_data_from_folder(folder_path):
        data = {}
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            file_key = os.path.splitext(filename)[0]  # Remove file extension to use as key
            try:
                # Attempt to read in UTF-8
                with open(filepath, 'r', encoding='utf-8') as file:
                    data[file_key] = file.read().strip()
            except UnicodeDecodeError:
                # Fallback to ISO-8859-1 or another encoding if UTF-8 fails
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
            
            # Check if the file has an image extension
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



    from datasets import load_dataset

    def format_data(sample):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"You are an AI assistant specialized in crochet knowledge. Generate original crochet pattern instructions.\n\n##IMAGE DESCRIPTION##: {sample['Image Description']}"
                        },
                        {
                            "type": "image",
                            "image": sample["image"],
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["Instruction"]}],
                },
            ]
        }

    # dataset_id = "philschmid/amazon-product-descriptions-vlm"
    # dataset = load_dataset(dataset_id, split="train")
    dataset = [format_data(sample) for sample in combined_data]


    import torch  
    from transformers import AutoProcessor

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
    processor = AutoProcessor.from_pretrained(model_id)
    model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable()


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
        texts = [processor(text=example["messages"][0]["content"][0]["text"], images=None, return_tensors="pt", padding=True) for example in examples]
        return texts

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning")

if __name__ == "__main__":
    with app.run():
        train_llama()