import modal

app = modal.App(name="llama-finetuning")

local_base_path = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset"
descriptions_path = f"{local_base_path}/descriptions"
images_path = f"{local_base_path}/images"
instructions_path = f"{local_base_path}/instructions"

@app.function(gpu="A100", timeout=900,
              secrets=[modal.Secret.from_name("my-huggingface-secret")],
              mounts=[
                  modal.Mount.from_local_dir(descriptions_path, remote_path="/root/dataset/descriptions"),
                  modal.Mount.from_local_dir(images_path, remote_path="/root/dataset/images"),
                  modal.Mount.from_local_dir(instructions_path, remote_path="/root/dataset/instructions"),
                  ],
                  image=modal.Image.debian_slim().run_commands("pip install --upgrade pip")
                  .pip_install(
                      "Pillow",  
                      "datasets",
                      "peft",
                      "torch",
                      "transformers>=4.30.0",
                      "unsloth==2024.10.0",
                      "unsloth-zoo",
                      "huggingface_hub>=0.26.2",
                      "qwen_vl_utils"
                      ))
def train_llama():
    import os
    from datasets import Dataset
    from PIL import Image
    import base64
    import io
    import torch  
    from transformers import AutoProcessor
    from unsloth import FastLanguageModel
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

    prompt = """You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

    When generating crochet instructions:
    1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
    2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
    3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
    4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
    5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
    6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

    ##IMAGE DESCRIPTION##: {image_description}"""

    system_message = "You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions."

    def format_data(sample):
        # print(sample["image"])
        return {"messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(image_description=sample["Image Description"]),
                    },{
                        "type": "image",
                        "image": sample["image"],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["Instruction"]}],
            },
        ],
        }

    # dataset_id = "philschmid/amazon-product-descriptions-vlm"
    # dataset = load_dataset(dataset_id, split="train")
    dataset = [format_data(sample) for sample in combined_data]


    import torch  
    from transformers import AutoProcessor
    from unsloth import FastLanguageModel

    from huggingface_hub import login
    login(token="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")
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # Initialize the model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    ## LoRA
    from peft import LoraConfig

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )

    ## Training Config
    from trl import SFTConfig
    args = SFTConfig(
        output_dir="fine-tuned-visionllama-unsloth",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=True,
        report_to="tensorboard",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    from qwen_vl_utils import process_vision_info
    from trl import SFTTrainer

    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = process_vision_info(example["messages"])[0]
        for example in examples:
            batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100
            batch["labels"] = labels

        return batch

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,  # Use the tokenizer from Unsloth
        peft_config=peft_config
    )

    # Apply Unsloth optimizations
    trainer = FastLanguageModel.get_peft_model(
        trainer,
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer.train()

    peft_model = trainer.model

    output_dir = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning"
    peft_model.save_pretrained(output_dir)

if __name__=="__main__":
    with stub.run():
        train_llama()