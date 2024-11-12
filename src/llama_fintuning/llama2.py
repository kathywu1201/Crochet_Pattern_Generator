import modal

app = modal.App(name="llama-finetuning")

local_base_path = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset"
descriptions_path = f"{local_base_path}/descriptions"
images_path = f"{local_base_path}/images"
instructions_path = f"{local_base_path}/instructions"
cache_dir_local = f"{local_base_path}/cache"


import os
os.makedirs(cache_dir_local, exist_ok=True)

# from transformers import TrainerCallback, TrainerState, TrainerControl
# import torch
# import logging
# import time

# class TokenCounterCallback(TrainerCallback):
#     def __init__(self):
#         super().__init__()
#         self.total_tokens = 0
#         self.current_epoch = 0
#         self.start_time = None

#     def on_train_begin(
#         self,
#         args,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         self.total_tokens = 0
#         self.start_time = time.time()
#         logging.info("Training started. Token counting initialized.")

#     def on_step_end(
#         self,
#         args,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         global CURRENT_BATCH_TOKENS
#         if CURRENT_BATCH_TOKENS > 0:
#             self.total_tokens += CURRENT_BATCH_TOKENS
#             elapsed_time = time.time() - self.start_time
#             tokens_per_second = self.total_tokens / elapsed_time if elapsed_time > 0 else 0
#             allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
#             reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # in GB
#             logging.info(f"Processed {CURRENT_BATCH_TOKENS} tokens. Total tokens so far: {self.total_tokens}")
#             logging.info(f"Training Speed: {tokens_per_second:.2f} tokens/sec")
#             logging.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
#             # Reset the counter
#             CURRENT_BATCH_TOKENS = 0

#     def on_epoch_end(
#         self,
#         args,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         self.current_epoch += 1
#         logging.info(f"Epoch {self.current_epoch} completed. Total tokens processed so far: {self.total_tokens}")


@app.function(gpu="H100:2", 
              timeout=7200,
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
                      ))
def train_llama():
    import os
    from datasets import Dataset
    from PIL import Image
    import base64
    import io
    import torch  
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
    from huggingface_hub import login
    import torch  
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    import logging
    import base64
    from pathlib import Path

    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    # Clear CUDA cache at the very start
    torch.cuda.empty_cache()

    # Set environment variable for PyTorch CUDA memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    def load_data():
        data = []
        for filename in os.listdir(image_description_folder):
            base_name = os.path.splitext(filename)[0]
            description_path = os.path.join(image_description_folder, filename)
            image_path = os.path.join(image_folder, f"{base_name}.png")  # Assuming images are .png
            instruction_path = os.path.join(instruction_folder, f"{base_name}.txt")

            if os.path.exists(description_path) and os.path.exists(image_path) and os.path.exists(instruction_path):
                try:
                    with open(description_path, 'r') as f:
                        description = f.read().strip()
                    with open(instruction_path, 'r') as f:
                        instruction = f.read().strip()
                    data.append({"description": description, "image_path": image_path, "instruction": instruction})
                except Exception as e:
                    logging.error(f"Error reading files for '{base_name}': {e}")
                    continue
        return Dataset.from_list(data)


    dataset = load_data()

    def encode_image(image_path):
        image = Image.open(image_path)
        return image

    prompt= """You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.  
                Create a detailed instruction on the provided ##IMAGE DESCRIPTION## and image.

                ##IMAGE DESCRIPTION##: {description}"""

    # from pathlib import Path
    def format_data(sample):
        encoded_image = encode_image(sample["image_path"])
        # print(encoded_image)
        dic = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt.format(description=sample['description']),
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
    dataset = [format_data(sample) for sample in dataset]
    print("dataset length", len(dataset))
    print(dataset[1]['messages'])



    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

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
        cache_dir="/root/.cache/huggingface",
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir="/root/.cache/huggingface")  # Use the same cache directory


    from peft import LoraConfig
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
    )

    from trl import SFTConfig

# Define SFTConfig with all necessary parameters
    sft_config = SFTConfig(
        output_dir="fine-tuned-visionllama",  # Directory to save and repository ID
        num_train_epochs=3,                     # Number of training epochs
        per_device_train_batch_size=4,          # Batch size per device during training
        gradient_accumulation_steps=16,          # Steps before performing a backward/update pass
        gradient_checkpointing=True,            # Use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # Optimizer
        logging_steps=5,                        # Log every 5 steps
        save_strategy="epoch",                  # Save checkpoint every epoch
        learning_rate=2e-4,                     # Learning rate
        bf16=True,                              # Use bfloat16 precision
        tf32=True,                              # Use tf32 precision
        max_grad_norm=0.3,                      # Max gradient norm
        warmup_ratio=0.03,                      # Warmup ratio
        lr_scheduler_type="constant",           # Learning rate scheduler
        push_to_hub=True,                       # Push model to hub
        report_to="tensorboard",                # Report metrics to TensorBoard
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Reentrant checkpointing
        dataset_text_field="",                  # Dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation
        max_seq_length=1024                     # Set max_seq_length to eliminate the warning
    )

    sft_config.remove_unused_columns=False

    from transformers import Qwen2VLProcessor
    from qwen_vl_utils import process_vision_info

    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        #image_inputs = [encode_image(example["image_path"]) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        ### Calculate number of tokens
        # batch_size, seq_length = batch["input_ids"].size()
        # num_tokens = batch_size * seq_length
        # if hasattr(torch.distributed, "get_rank"):  # Check if distributed training is active
        #     rank = torch.distributed.get_rank()
        # else:
        #     rank = 0
        # global CURRENT_BATCH_TOKENS
        # CURRENT_BATCH_TOKENS = num_tokens

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
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

        # print("batch", batch)
        return batch


    from trl import SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning")

if __name__ == "__main__":
    with app.run():
        train_llama()