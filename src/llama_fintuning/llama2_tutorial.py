import modal

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
    from datasets import Dataset, load_dataset
    from PIL import Image
    import base64
    import io
    import torch  
    from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
    from huggingface_hub import login
    import torch  
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor
    import logging
    import base64
    from pathlib import Path
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from qwen_vl_utils import process_vision_info

    # Hugging Face login

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")


    # Paths for images and data
    prompt= """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
    Only return description. The description should be SEO optimized and for a better mobile search experience.

    ##PRODUCT NAME##: {product_name}
    ##CATEGORY##: {category}"""
    # Helper function to read files and map by filename

    # Convert dataset to OAI messages
    def format_data(sample):
        return {"messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.format(product_name=sample["Product Name"], category=sample["Category"]),
                            },{
                                "type": "image",
                                "image": sample["image"],
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": sample["description"]}],
                    },
                ],
            }

    # Load dataset from the hub
    dataset_id = "philschmid/amazon-product-descriptions-vlm"
    dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")

    # Convert dataset to OAI messages
    # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
    dataset = [format_data(sample) for sample in dataset]

    print(dataset[345]["messages"])

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

#   BitsAndBytesConfig int-4 config   
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    text = processor.apply_chat_template(
        dataset[2]["messages"], tokenize=False, add_generation_prompt=False)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",)

    args = SFTConfig(
        output_dir="fine-tuned-visionllama", # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=4,          # batch size per device during training
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=5,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
        dataset_text_field="", # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
    )
    args.remove_unused_columns=False

    def collate_fn(examples):
    # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

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

        return batch


    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        dataset_text_field="", # needs dummy value
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    with app.run():
        train_llama()