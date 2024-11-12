import modal

from pathlib import Path

app = modal.App(name="llama-finetuning")

local_base_path = "/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/dataset"
descriptions_path = f"{local_base_path}/descriptions"
images_path = f"{local_base_path}/images"
instructions_path = f"{local_base_path}/instructions"
cache_dir_local = f"{local_base_path}/cache"

import os
os.makedirs(cache_dir_local, exist_ok=True)

@app.function(gpu="A100", timeout=3600,
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
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
    from huggingface_hub import login
    import torch  
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    import logging
    from pathlib import Path
    from qwen_vl_utils import process_vision_info

    # Hugging Face login

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

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

    import base64
    def encode_image(image_path):
        """
        Encodes an image to a base64 string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"<|image|>data:image/{Path(image_path).suffix[1:]};base64,{encoded_string}"
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {e}")
            return ""
        
#######testing encode_image function#######
    def test_encode_image():
        sample_image_path = "/root/dataset/images/ALK0125-025285M.png"  # Replace with a valid path
        encoded = encode_image(sample_image_path)
        print(encoded)
    # Run the test
    test_encode_image()
#######testing encode_image function#######

    def format_data(sample, include_image=True, embed_image=False):
        """
        Formats a data sample according to Llama's prompt templates.

        Args:
            sample (dict): A single data sample containing 'description', 'image_path', and 'instruction'.
            include_image (bool): Whether to include the image in the prompt.
            embed_image (bool): Whether to embed the image data as a Base64 string.

        Returns:
            dict: Formatted messages suitable for Llama model.
        """
        messages = []

        if include_image:
            # When including an image, do NOT add a system message
            if embed_image:
                image_content = encode_image(sample['image_path'])  # Function to encode image
            else:
                image_content = f"<|image|>{sample['image_path']}"  # Using image path

            messages.append({
                "role": "user",
                "content": f"<|start_header_id|>user<|end_header_id|>\n"
                           f"{image_content} Describe this image in two sentences.<|eot_id|>"
            })
        else:
            # When NOT including an image, add a system message
            messages.append({
                "role": "system",
                "content": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                           "You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.<|eot_id|>"
            })
            messages.append({
                "role": "user",
                "content": f"<|start_header_id|>user<|end_header_id|>\n"
                           f"You are an AI assistant specialized in crochet knowledge. Generate original crochet pattern instructions.\n\n##IMAGE DESCRIPTION##: {sample['description']}<|eot_id|>"
            })

        # Always include the assistant's response
        messages.append({
            "role": "assistant",
            "content": f"<|start_header_id|>assistant<|end_header_id|>\n"
                       f"{sample['instruction']}<|eot_id|>"
        })

        return {
            "messages": messages
        }

    dataset = dataset.map(
        lambda sample: format_data(sample, include_image=True, embed_image=False),  # Adjust parameters as needed
        remove_columns=["description", "image_path", "instruction"]
    )
##############test load_image function#############
    def verify_image_paths(dataset):
        """
        Verifies that all image paths in the dataset exist.
        
        Args:
            dataset (Dataset): The Hugging Face dataset.
        
        Raises:
            FileNotFoundError: If any image path does not exist.
        """
        missing_images = []
        for idx, sample in enumerate(dataset):
            image_path = sample['image_path']
            if not os.path.exists(image_path):
                missing_images.append((idx, image_path))
        
        if missing_images:
            for idx, path in missing_images:
                logging.error(f"Sample {idx} has a missing image at path: {path}")
            raise FileNotFoundError(f"{len(missing_images)} images are missing. Check the logs for details.")
        else:
            logging.info("All image paths are valid and exist.")
    #Run test
    dataset = load_data()
    verify_image_paths(dataset)


    def verify_training_data(dataset):
        """
        Verifies that each sample in the dataset has a valid image reference.
        Logs samples with invalid or missing images.
        """
        print(dataset)
        for idx, sample in enumerate(dataset):
            try:
                img_input = process_vision_info(sample["messages"])[0]
                if img_input is None:
                    logging.warning(f"Sample {idx}: No image reference found.")
                    continue
                if isinstance(img_input, str):
                    if img_input.startswith("<|image|>data:image/"):
                        base64_str = img_input.split("base64,", 1)[1].split(" ", 1)[0]
                        try:
                            base64.b64decode(base64_str)
                        except Exception as e:
                            logging.warning(f"Sample {idx}: Invalid base64 string: {e}")
                    elif img_input.startswith("<|image|>"):
                        image_path = img_input[len("<|image|>"):]
                        if not os.path.exists(image_path):
                            logging.warning(f"Sample {idx}: Image path does not exist: {image_path}")
                    else:
                        logging.warning(f"Sample {idx}: Unrecognized image input format: {img_input}")
            except Exception as e:
                logging.error(f"Sample {idx}: Error during verification: {e}")
    verify_training_data(dataset)
        

#     model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

#         # BitsAndBytesConfig int-4 config
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )

#     # Load model and tokenizer
#     model = AutoModelForVision2Seq.from_pretrained(
#         model_id,
#         cache_dir="/root/.cache/huggingface",
#         device_map="auto",
#         # attn_implementation="flash_attention_2", # not supported for training
#         torch_dtype=torch.bfloat16,
#         quantization_config=bnb_config
#     )
#     processor = AutoProcessor.from_pretrained(model_id, cache_dir="/root/.cache/huggingface")  # Use the same cache directory


#     from peft import LoraConfig
#     # LoRA config based on QLoRA paper & Sebastian Raschka experiment
#     peft_config = LoraConfig(
#             lora_alpha=16,
#             lora_dropout=0.05,
#             r=8,
#             bias="none",
#             target_modules=["q_proj", "v_proj"],
#             task_type="CAUSAL_LM",
#     )

#     from trl import SFTConfig
#     from trl import SFTConfig

# # Define SFTConfig with all necessary parameters
#     sft_config = SFTConfig(
#         output_dir="fine-tuned-visionllama",  # Directory to save and repository ID
#         num_train_epochs=3,                     # Number of training epochs
#         per_device_train_batch_size=4,          # Batch size per device during training
#         gradient_accumulation_steps=8,          # Steps before performing a backward/update pass
#         gradient_checkpointing=True,            # Use gradient checkpointing to save memory
#         optim="adamw_torch_fused",              # Optimizer
#         logging_steps=5,                        # Log every 5 steps
#         save_strategy="epoch",                  # Save checkpoint every epoch
#         learning_rate=2e-4,                     # Learning rate
#         bf16=True,                              # Use bfloat16 precision
#         tf32=True,                              # Use tf32 precision
#         max_grad_norm=0.3,                      # Max gradient norm
#         warmup_ratio=0.03,                      # Warmup ratio
#         lr_scheduler_type="constant",           # Learning rate scheduler
#         push_to_hub=True,                       # Push model to hub
#         report_to="tensorboard",                # Report metrics to TensorBoard
#         gradient_checkpointing_kwargs={"use_reentrant": False},  # Reentrant checkpointing
#         dataset_text_field="",                  # Dummy field for collator
#         dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation
#         max_seq_length=1024                     # Set max_seq_length to eliminate the warning
#     )

#     sft_config.remove_unused_columns=False

#     from transformers import Qwen2VLProcessor
#     from qwen_vl_utils import process_vision_info

    # def collate_fn(examples):
    #     """
    #     Collate function to process batch examples.
        
    #     Args:
    #         examples (list): A list of examples from the dataset.
        
    #     Returns:
    #         dict: A batch dictionary with tokenized texts, processed images, and labels.
    #     """
    #     # Extract texts from messages
    #     texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        
    #     # Extract image inputs
    #     image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
        
    #     # Process images
    #     loaded_images = []
    #     for img_input in image_inputs:
    #         if isinstance(img_input, str):
    #             if img_input.startswith("<|image|>data:image/"):
    #                 # It's a base64-encoded image
    #                 try:
    #                     # Extract the base64 string
    #                     base64_str = img_input.split("base64,", 1)[1]
    #                     image_data = base64.b64decode(base64_str)
    #                     image = Image.open(io.BytesIO(image_data)).convert("RGB")
    #                 except Exception as e:
    #                     logging.error(f"Error decoding base64 image: {e}")
    #                     # Create a blank image or handle as needed
    #                     image = Image.new("RGB", (224, 224))
    #             elif img_input.startswith("<|image|>"):
    #                 # It's an image path
    #                 image_path = img_input[len("<|image|>"):]
    #                 try:
    #                     image = Image.open(image_path).convert("RGB")
    #                 except Exception as e:
    #                     logging.error(f"Error loading image from path {image_path}: {e}")
    #                     # Create a blank image or handle as needed
    #                     image = Image.new("RGB", (224, 224))
    #             else:
    #                 logging.error(f"Unrecognized image input format: {img_input}")
    #                 image = Image.new("RGB", (224, 224))
    #         elif isinstance(img_input, Image.Image):
    #             # It's already a PIL Image
    #             image = img_input
    #         else:
    #             logging.error(f"Invalid image input type: {type(img_input)}")
    #             image = Image.new("RGB", (224, 224))
            
    #         loaded_images.append(image)
        
    #     # Tokenize texts and process images
    #     try:
    #         batch = processor(text=texts, images=loaded_images, return_tensors="pt", padding=True)
    #     except Exception as e:
    #         logging.error(f"Error during processor call: {e}")
    #         raise e
        
    #     # Process labels
    #     labels = batch["input_ids"].clone()
    #     labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding tokens
        
    #     # Ignore image token indices in loss computation
    #     if isinstance(processor, Qwen2VLProcessor):
    #         image_tokens = [151652, 151653, 151655]
    #     else:
    #         image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        
    #     for image_token_id in image_tokens:
    #         labels[labels == image_token_id] = -100
        
    #     batch["labels"] = labels
        
    #     return batch

    # from trl import SFTTrainer
    # trainer = SFTTrainer(
    #     model=model,
    #     args=sft_config,
    #     train_dataset=dataset,
    #     data_collator=collate_fn,
    #     peft_config=peft_config,
    #     tokenizer=processor.tokenizer,
    # )

    # # Train the model
    # trainer.train()

#     # Save the model
#     model.save_pretrained("/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning")

if __name__ == "__main__":
    with app.run():
        train_llama()