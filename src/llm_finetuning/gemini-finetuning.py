import os
import argparse
import time
from google.cloud import storage
import vertexai
from vertexai.preview.tuning import sft
from vertexai.generative_models import GenerativeModel

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
bucket_name = os.environ["GCS_BUCKET_NAME"]
TRAIN_DATASET = "gs://crochet-patterns/train.jsonl"
VALIDATION_DATASET = "gs://crochet-patterns/test.jsonl"
GCP_LOCATION = "us-central1"
GENERATIVE_SOURCE_MODEL = "gemini-1.5-flash-002"  # Use the desired model version

# Configuration for content generation
generation_config = {
    "max_output_tokens": 3000,
    "temperature": 0.75,
    "top_p": 0.95,
}



SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

When generating crochet instructions:
1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.
"""


# Initialize Vertex AI with the project and location
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# Fine-tuning function
def train(wait_for_job=False):
    print("Starting model fine-tuning...")

    # Supervised Fine Tuning (SFT)
    sft_tuning_job = sft.train(
        source_model=GENERATIVE_SOURCE_MODEL,
        train_dataset=TRAIN_DATASET,
        validation_dataset=VALIDATION_DATASET,
        epochs=5,  # Adjust based on your needs
        adapter_size=4,
        learning_rate_multiplier=1.0,
        tuned_model_display_name="custom-crochet-pattern-model-v2",
    )
    print("Training job started. Monitoring progress...")

    # Optionally wait and refresh the job status
    time.sleep(60)
    sft_tuning_job.refresh()

    if wait_for_job:
        print("Checking status of tuning job:")
        while not sft_tuning_job.has_ended:
            time.sleep(60)
            sft_tuning_job.refresh()
            print("Job in progress...")

    print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
    print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")
    print(f"Experiment details: {sft_tuning_job.experiment}")

# Function to test the fine-tuned model with a chat-like interface
def chat():
    print("Testing the fine-tuned model with a sample prompt...")

    # Set the endpoint of the fine-tuned model
    MODEL_ENDPOINT = "projects/376381333238/locations/us-central1/endpoints/3614500440290361344"

    # Load the fine-tuned model
    generative_model = GenerativeModel(MODEL_ENDPOINT, system_instruction=[SYSTEM_INSTRUCTION])

    # Sample query prompt to generate pattern instructions
    query = "Give me an instruction of how to crochet a heart shaped coaster with six rounds"
    print("Input prompt:", query)

    # Generate content from the fine-tuned model
    response = generative_model.generate_content(
        [query],  # Input prompt
        generation_config=generation_config,  # Configuration settings
        stream=False,  # Disable streaming
    )
    generated_text = response.text
    print("Fine-tuned LLM Response:", generated_text)

# Main function to handle CLI arguments for training or chatting
def main(args=None):
    print("CLI Arguments:", args)

    if args.train:
        train(wait_for_job=True)  # Set to True to wait until the job completes

    if args.chat:
        chat()

if __name__ == "__main__":
    # Generate the input arguments parser
    parser = argparse.ArgumentParser(description="CLI for fine-tuning and testing the Gemini model")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with the fine-tuned model",
    )

    args = parser.parse_args()
    main(args)
