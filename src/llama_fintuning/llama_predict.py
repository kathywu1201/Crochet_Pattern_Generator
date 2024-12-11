import modal
# from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel, PeftConfig
# from PIL import Image
# import torch
# import os

# Define Modal app
app = modal.App(name="llama-predict")

# Configure secrets outside of the function
huggingface_secret = modal.Secret.from_name("my-huggingface-secret")

# Paths for test images and description files
test_images_path = "test/images"
test_descriptions_path = "test/image_descriptions"

@app.function(gpu="H100", timeout=8000, 
              secrets=[huggingface_secret], mounts=[
    modal.Mount.from_local_dir(test_images_path, remote_path="/root/images"),
    modal.Mount.from_local_dir(test_descriptions_path, remote_path="/root/descriptions"),
    ],
              image=modal.Image.debian_slim().pip_install(
                      "Pillow",  
                      "torch==2.4.0",
                      "transformers==4.45.1",
                      "huggingface_hub>=0.26.2",
                      "peft",
                      "accelerate>=0.26.0",
                      "trl"
                      ))
def run_prediction():
    from huggingface_hub import login
    from transformers import AutoProcessor, MllamaForConditionalGeneration
    from peft import PeftModel
    import torch
    import os
    from PIL import Image
    # from trl import setup_chat_format

    # Ensure Hugging Face token is available as an environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        raise ValueError("Hugging Face token not found. Set the HF_TOKEN environment variable.")

    # UNCOMMENT THE FOLLOWING IF USE BASE MODEL AND ADAPTER """"

    # base_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # adapter_model_id = "ccwxp116/fine-tuned-visionllama"  # Adapter model on Hugging Face

    # # Load base model and tokenizer
    # base_model_reload = MllamaForConditionalGeneration.from_pretrained(
    #     base_model_id,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    # processor = AutoProcessor.from_pretrained(base_model_id)

    # model = PeftModel.from_pretrained(base_model_reload, adapter_model_id)

    # model = model.merge_and_unload()

    # """"

    # UNCOMMENT THE FOLLOWING IF USE MERGED MODEL """"

    base_model = "ccwxp116/fine-tuned-visionllama-v1"

    processor = AutoProcessor.from_pretrained(base_model)

    model = MllamaForConditionalGeneration.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # """"

    # Load test image
    image_path = "/root/images/heart.png"  
    image = Image.open(image_path)
    print("image:", image)

    # Load description from a text file
    description_path = "/root/descriptions/heart.txt"  
    with open(description_path, "r") as f:
        description = f.read().strip()

    # Define the prompt template and format it with the description
    # prompt = f"""
    # You are a crochet expert with deep knowledge of patterns, materials, and techniques. Your task is to provide a complete set of crochet instructions based on the provided image and following image description, covering every detail necessary for someone to recreate the piece accurately. 

    # Please include:
    # 1. **Materials Needed**: Specify the type of yarn, recommended yarn weight, suggested colors, and any additional tools (e.g., crochet hook size, scissors).
    # 2. **Color and Yarn Details**: Describe the colors used and suggest specific shades or brands, if applicable.
    # 3. **Step-by-Step Instructions**: Provide detailed, easy-to-follow steps, starting with the foundation chain and covering each row or round in sequence. Include stitch types, counts, and any color changes or special techniques.

    # Here is the image description to base your instructions on:

    # ## IMAGE DESCRIPTION ##
    # {description}
    # """
        
    prompt = f"""
    You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the provided image and following image description, covering every detail necessary for someone to recreate the piece accurately, using your expertise in crochet. 

    When generating crochet instructions:
    1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
    2. You are not limited to summarizing the provided text chunks. Instead, use them as background information to inform your crochet expertise.
    3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
    4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
    5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
    6. Do not summarize content from the chunks unless explicitly asked to; your primary goal is to generate new instructions.

    You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.
    Here is the image description to base your instructions on:

    ## IMAGE DESCRIPTION ##
    {description}
    """

    # Format message and input for processing
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]

    print("message:", messages)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(image, prompt, return_tensors='pt').to(model.device)

    output = model.generate(**inputs, max_new_tokens=2000)

    print(processor.decode(output[0]))

    # UNCOMMENT THE FOLLOWING IF WANT TO SAVE THE FULL MODEL
    
    # new_model = "llama-3.2-vision-finetuned-new"

    # model.save_pretrained(new_model)
    # processor.save_pretrained(new_model)

    # model.push_to_hub(new_model, use_temp_dir=False)
    # processor.push_to_hub(new_model, use_temp_dir=False)

    # """"

# Run the prediction function in Modal
if __name__ == "__main__":
    with app.run():
        run_prediction()




