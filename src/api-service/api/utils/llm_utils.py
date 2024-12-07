import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
import io
from PIL import Image
from pathlib import Path
import traceback
from vertexai.generative_models import GenerativeModel, ChatSession, Part

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
# GCP_PROJECT = "crochet-ai"
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-1.5-flash-002"
MODEL_ENDPOINT = "projects/376381333238/locations/us-central1/endpoints/3614500440290361344"

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 2000,  # Maximum number of tokens for output
    "temperature": 0.1,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are a highly skilled AI assistant specialized in creating crochet patterns. 
Your task is to generate a detailed crochet pattern for the product shown in the image. 
However, you must prioritize user preferences from their input when they differ from the image. 
For example, if the user specifies a "red crochet product" but the image shows a "blue crochet product," adjust the pattern accordingly (e.g., update the color of the yarn in the "Materials Needed" section).

Your response must strictly follow the format below, ensuring the number of rounds or steps matches the typical requirements for the product. Do not generate more rounds or steps than necessary to complete the project. Avoid unnecessary line breaks or redundant information to ensure the output is clear, organized, and precise.

Here is the format:

**Product Description:**
- Provide a brief description of the crochet product. If the user specifies preferences (e.g., color, size), adjust the description accordingly.

**Materials Needed:**
- Yarn type and weight: [Specify the type(s), weight(s), and color(s), adjusted based on user input if necessary.]
- Crochet hook size: [Include the recommended hook size.]
- Additional tools: [List any additional tools, such as scissors or tapestry needles.]

**Abbreviations:**
- Include common crochet abbreviations used in the pattern (e.g., sc = single crochet, ch = chain, etc.). Add others as needed.

**Pattern Instructions:**
- Provide detailed step-by-step instructions.
- Include row/round numbers, stitch counts, repeats, and any special techniques.
- Ensure each step is on its own line without excessive line breaks.
- Use clear and concise language to guide the user through the crochet process.
"""

generative_model = GenerativeModel(
	MODEL_ENDPOINT,
	system_instruction=[SYSTEM_INSTRUCTION]
)

DESCRIPTION_PROMPT = '''
You are an expert in textile arts with a specialization in crochet. 
Your task is to analyze the provided image of a crochet object and generate a detailed description focusing exclusively on the intricate details of the crochet work. 
Describe the crochet product shown in this image, focusing on its pattern and texture. 
Ignore any other objects or elements in the image.
The description should encompass the following aspects:

Number of Threads:
Determine and specify the total number of threads used in creating the crochet object.
If possible, provide information on the thickness or gauge of the threads.
Stitch Types:
Identify and list all the types of stitches present in the crochet piece (e.g., single crochet, double crochet, half-double crochet, treble crochet, etc.).
Describe any unique or complex stitch patterns utilized.
Yarn Color:
Accurately describe the color(s) of the yarn used.
Mention any color variations, gradients, or patterns resulting from the yarn colors.
Knit vs. Crochet Distinction:
Analyze the construction of the object to determine whether it is knitted or crocheted.
Provide reasoning for the distinction, highlighting specific features that indicate crochet techniques over knitting, or vice versa.
Overall Texture:
Describe the overall texture of the crochet piece, focusing on the surface appearance and feel.
Size and Dimension:
If possible, estimate the size and dimensions of the crochet object.
Additional Guidelines:
Focus Exclusively on Crochet Details: Ensure that the description remains concentrated on the crochet aspects mentioned above. Avoid general comments about the object's appearance, functionality, or aesthetics unless they directly relate to the crochet techniques or materials used.
Clarity and Precision: Use clear and precise language to convey each detail. Where measurements or counts are involved, provide them in appropriate units (e.g., number of threads, specific row counts).
Technical Accuracy: Ensure that all crochet terminology and descriptions are technically accurate, reflecting a deep understanding of crochet methods and practices.
Structured Format: Present the information in a well-organized manner, possibly using bullet points or numbered lists for each of the five key aspects to enhance readability.
'''

# Keep the existing Gemini model for image description
description_model = GenerativeModel(
    "gemini-1.5-flash-002",
    system_instruction=[DESCRIPTION_PROMPT]
)

# Initialize chat sessions
chat_sessions: Dict[str, ChatSession] = {}

def create_chat_session() -> ChatSession:
    """Create a new chat session with the model"""
    return generative_model.start_chat()

def generate_chat_response(chat_session: ChatSession, message: Dict) -> str:
    """
    Generate a response using the chat session to maintain history.
    Handles both text and image inputs.
    
    Args:
        chat_session: The Vertex AI chat session
        message: Dict containing 'content' (text) and optionally 'image' (base64 string)
    
    Returns:
        str: The model's response
    """
    # response = chat_session.send_message(
    #     message,
    #     generation_config=generation_config
    # )
    # return response.text
    try:
        # Initialize parts list for the message
        message_parts = []
        
        
        # Process image if present
        if message.get("image"):
            try:
                # Extract base64 data and mime type
                base64_string = message.get("image")
                if ',' in base64_string:
                    header, base64_data = base64_string.split(',', 1)
                    mime_type = header.split(':')[1].split(';')[0]
                else:
                    base64_data = base64_string
                    mime_type = 'image/jpeg'
                
                image_bytes = base64.b64decode(base64_data)
                image_part = Part.from_data(image_bytes, mime_type=mime_type)

                
                # First call to generate description
                description_parts = [
                    image_part,
                    "Please analyze this crochet item and provide a detailed description of the crochet item."
                ]
                description_response = description_model.start_chat().send_message(
                    description_parts,
                    generation_config=generation_config
                )
                generated_description = description_response.text

                # Step 2: Generate crochet instructions using both image and description
                instruction_prompt = f"""
                Based on this image and the following description:

                {generated_description}

                {message.get('content', 'Please provide detailed crochet instructions for recreating this item.')}
                """
                
                # description-based prompt
                message_parts = instruction_prompt
                
            except ValueError as e:
                print(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

        if not message_parts:
            raise ValueError("Message must contain either text content or image")

        # print("Message parts:", message_parts)
        # Send message with all parts to the model
        response = generative_model.generate_content(
            [message_parts],  
            generation_config=generation_config, 
            stream=False, 
        )

        return response.text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )

def rebuild_chat_session(chat_history: List[Dict]) -> ChatSession:
    """Rebuild a chat session with complete context"""
    new_session = create_chat_session()
    
    for message in chat_history:
        if message["role"] == "user":
            generate_chat_response(new_session, message)
        # 
        #     response = new_session.send_message(
        #         message["content"],
        #         generation_config=generation_config
        #     )
    
    return new_session