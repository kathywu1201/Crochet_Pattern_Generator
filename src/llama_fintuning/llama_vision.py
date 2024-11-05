# from datasets import Dataset


# data_prompt = """You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

# When generating crochet instructions:
# 1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
# 2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
# 3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
# 4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
# 5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
# 6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

# You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.

# ### Input:
# {}
# ### Response:
# {}"""

# system_massage -"You are an AI assistant specialized in crochet knowledge."


# src/llm-finetuning/llama_finetuning.py

from transformers import LlamaForCausalLM, LlamaProcessor, Trainer, TrainingArguments
import torch
from torchvision import transforms
from PIL import Image
import json
from google.cloud import storage
from torch.utils.data import Dataset
from PIL import Image
import os
GCP_PROJECT = os.environ["GCP_PROJECT"]

# Load model and processor
model = LlamaForCausalLM.from_pretrained("Meta/Llama-3.2-vision")
processor = LlamaProcessor.from_pretrained("Meta/Llama-3.2-vision")

# Define data transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
def load_data(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def preprocess_data(example):
    # Load and transform image
    image = Image.open(example["image_path"]).convert("RGB")
    image = transform(image)

    # Prepare input and output for Llama 3.2 Vision
    inputs = processor(text=example["prompt"], images=image, return_tensors="pt")
    outputs = processor(text=example["response"], return_tensors="pt")["input_ids"]
    return {"inputs": inputs, "labels": outputs}

# Load and preprocess the data
train_data = load_data("/path/to/your/data/train.jsonl")
train_data = [preprocess_data(example) for example in train_data]

# Training configuration
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Fine-tune the model
trainer.train()

