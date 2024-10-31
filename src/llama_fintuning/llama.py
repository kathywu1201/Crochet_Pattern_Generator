import os
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

from google.cloud import storage

#setup GCP connection:
GCP_PROJECT = os.environ["GCP_PROJECT"]
data = pd.read_json("/Users/ciciwxp/Desktop/AC215/Crochet_Pattern_Generator/src/llama_fintuning/transformed_file.json", lines = True)
# data = pd.read_json("gs://crochet-patterns-bucket/training/image_descriptions_jsonl/train.jsonl", lines = True)  # Replace with your dataset path

# VALIDATION_DATASET = "gs://crochet-patterns-bucket/training/image_descriptions_jsonl/test.jsonl"
 
# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
# Warnings
import warnings
warnings.filterwarnings("ignore")


max_seq_length = 5020
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)
 
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)
print(model.print_trainable_parameters())

data_prompt = """You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

When generating crochet instructions:
1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.

### Input:
{}
### Response:
{}"""
 
EOS_TOKEN = tokenizer.eos_token
def formatting_prompt(examples):
    inputs       = examples["context"]
    outputs      = examples["response"]
    texts = []
    for input_, output in zip(inputs, outputs):
        text = data_prompt.format(input_, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


training_data = Dataset.from_pandas(filtered_data)
training_data = training_data.map(formatting_prompt, batched=True)

## start training: 
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=40,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)
 
trainer.train()