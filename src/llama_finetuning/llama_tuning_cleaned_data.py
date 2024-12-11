import modal
import os

app = modal.App(name="llama-finetuning")

local_base_path = "dataset"
images_path = f"{local_base_path}/images"
cache_dir_local = f"{local_base_path}/cache"
filtered_dataset_path = f"{local_base_path}/filtered_dataset.json"

# descriptions_path = f"{local_base_path}/descriptions"
# instructions_path = f"{local_base_path}/instructions"

os.makedirs(cache_dir_local, exist_ok=True)

@app.function(gpu="H100:2", 
              timeout=72000,
              secrets=[
                  modal.Secret.from_name("my-huggingface-secret"),
                  modal.Secret.from_name("wandb-secret")
                  ],
              mounts=[
                  modal.Mount.from_local_file(filtered_dataset_path, remote_path="/root/dataset/filtered_dataset.json"),
                   modal.Mount.from_local_dir(images_path, remote_path="/root/dataset/images"),
                  modal.Mount.from_local_dir(cache_dir_local, remote_path="/root/.cache/huggingface")
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
                      "tensorboard",
                      "wandb",
                      "scikit-learn"
                      ))
def train_llama():
    # Imports
    import os
    import json
    import torch 
    import wandb
    import logging
    from PIL import Image
    from datasets import Dataset, DatasetDict 
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback, Qwen2VLProcessor
    from huggingface_hub import login 
    # from sklearn.model_selection import train_test_split
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig
    from qwen_vl_utils import process_vision_info

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    # Clear CUDA cache at the very start
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Details about CUDA GPU
    os.system('nvidia-smi')

    # Hugging Face login
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")
    login(token=token)
    
    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("wandb-secret not found. Set the wandb-secret environment variable.")
    wandb.login(key=wandb_api_key)
    wandb.init(
        project="Crochet-AI-v3",
        config={
            "learning_rate": 2e-4,
            "epochs": 5,
            "batch_size": 2,
            "gradient_accumulation_steps": 32
        }
    )

    # Load dataset
    def load_filtered_dataset(path):
        with open(path, "r") as infile:
            dataset_dict = json.load(infile)
        return Dataset.from_dict(dataset_dict)

    dataset = load_filtered_dataset(filtered_dataset_path)

    def encode_image(image_path):
        image = Image.open(image_path)
        return image

    system_prompt = """
    You are a highly skilled AI assistant specializing in creating step-by-step crochet patterns. 
    Your goal is to generate a clear, well-structured, and detailed crochet pattern based on the user input.

    If the user uploads an image, derive the pattern from the provided image and any special instruction given by the user.  
    If no image is uploaded, generate the pattern based on the user-provided text description.  

    Although your training data may sometimes include unstructured or messy content, always ensure that your response adheres to the following format:

    1. **Materials Needed**:
    - Specify the yarn type, weight, recommended colors, crochet hook size, and any additional tools required.
    2. **Abbreviations** (if applicable):
    - Provide a clear list of any abbreviations used in the pattern (e.g., sc = single crochet, ch = chain).
    3. **Pattern Instructions**:
    - Deliver detailed, step-by-step instructions tailored for both beginners and experts.
    - Clearly indicate the number of stitches, rows, or rounds required at each step.
    - Highlight any special techniques or unique stitches used to replicate the product.

    Even if the provided training data is messy or inconsistent, you must learn to output responses in a polished, clear, and structured manner.
    """


    def format_data(sample):
        encoded_image = encode_image(sample["image_path"])
        # print(encoded_image)
        dic = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a highly skilled AI assistant specializing in creating crochet patterns. Your role is to analyze user inputs, including text and images of crochet products, and provide detailed, accurate, and well-structured responses. Always respond in a clear and organized format, including sections such as Crochet Product Name, Materials, Measurements, Abbreviations, and Instructions. Adapt your responses based on user queries, providing only the requested information when specified or a complete instruction set if not."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a complete instruction for this crochet product.",
                        },
                        {
                            "type": "image",
                            "image": encoded_image,  # Using the base64 encoded image
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": sample["instruction"]
                        }
                    ],
                },
            ]
        }
        # print(">>>>formatted dictionary", dic)
        return dic


    # dataset_id = "philschmid/amazon-product-descriptions-vlm"
    # dataset = load_dataset(dataset_id, split="train")
    # dataset = [format_data(sample) for sample in dataset]
    # print("dataset length", len(dataset))
    # print(dataset[1]['messages'])

    # Preprocess dataset
    formatted_dataset = [format_data(sample) for sample in dataset]
    print("dataset length", len(formatted_dataset))
    print(formatted_dataset[1]['messages'])

    # # rows = formatted_dataset.to_list()

    # # Split dataset into train and eval
    # train_data, eval_data = train_test_split(formatted_dataset, test_size=0.2, random_state=42)
    # dataset_dict = DatasetDict({
    #     "train": Dataset.from_list(train_data),
    #     "eval": Dataset.from_list(eval_data)
    # })

    # print("train dataset example:", dataset_dict["train"][1]['messages'])
    # print("val dataset example:", dataset_dict["eval"][1]['messages'])

    # Get the total number of samples
    total_samples = len(formatted_dataset)

    # Calculate the split indices
    train_size = int(0.85 * total_samples)  # First 80% for training

    # Split the dataset manually
    train_dataset = formatted_dataset[:train_size]  # First 80%
    val_dataset = formatted_dataset[train_size:]  # Last 20%

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "eval": val_dataset
    })

    # Debugging: Check the structure of the train and validation datasets
    print(f"Train dataset size: {len(dataset_dict['train'])}")
    print(f"Validation dataset size: {len(dataset_dict['eval'])}")
    print(f"Train dataset example: {dataset_dict['train'][0]['messages']}")
    print(f"Validation dataset example: {dataset_dict['eval'][0]['messages']}")

##### models ########

    # Model and processor
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    new_model="fine-tuned-visionllama-v3"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir="/root/.cache/huggingface")  # Use the same cache directory
    
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
    )

    # 32, 8, 0.1

    # Define SFTConfig with all necessary parameters
    sft_config = SFTConfig(
        output_dir=new_model,                   # Directory to save and repository ID
        num_train_epochs=5,                     # Number of training epochs
        per_device_train_batch_size=2,          # Batch size per device during training
        per_device_eval_batch_size=2,           # Evaluation batch size per device
        gradient_accumulation_steps=32,         # Steps before performing a backward/update pass
        gradient_checkpointing=True,            # Use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # Optimizer
        logging_steps=1,                       # Log every 5 steps
        save_strategy="epoch",                  # Save checkpoint every epoch
        evaluation_strategy="epoch",            # Evaluate after every epoch
        load_best_model_at_end=True,            # Load best model at the end
        learning_rate=2e-4,                     # Learning rate
        bf16=True,                              # Use bfloat16 precision
        tf32=True,                              # Use tf32 precision
        max_grad_norm=0.3,                      # Max gradient norm
        warmup_ratio=0.03,                      # Warmup ratio
        lr_scheduler_type="constant",           # Learning rate scheduler
        push_to_hub=True,                       # Push model to hub
        report_to=["tensorboard", "wandb"],                # Report metrics to TensorBoard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Reentrant checkpointing
        dataset_text_field="",                  # Dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation
        max_seq_length=2048                     # Set max_seq_length to eliminate the warning
    )

    sft_config.remove_unused_columns=False

    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652,151653,151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["eval"],
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
        callbacks=[early_stopping_callback]
    )

    # Train the model
    trainer.train()
    # Save the model

    model.save_pretrained(new_model)
    processor.save_pretrained(new_model)
    model.push_to_hub(new_model, use_temp_dir=False)
    processor.push_to_hub(new_model, use_temp_dir=False)

    wandb.finish()

if __name__ == "__main__":
    with app.run():
        train_llama()