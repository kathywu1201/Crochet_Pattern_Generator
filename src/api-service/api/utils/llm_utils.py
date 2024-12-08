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
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-1.5-flash-002"

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 3000,  # Maximum number of tokens for output
    "temperature": 0.1,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in crochet knowledge. Your primary task is to generate original crochet pattern instructions based on the user's prompt, using your expertise in crochet. 

When generating crochet instructions:
1. Focus on creating a new pattern or providing instructions based on the specific item mentioned in the user's prompt.
3. Prioritize crafting clear, step-by-step pattern instructions, including stitch types, materials, and any special techniques, as appropriate for the item in the prompt.
4. If the provided chunks do not offer enough information to generate a full pattern, fill in the gaps with plausible crochet knowledge based on common techniques.
5. Ensure that your responses are creative and provide detailed crochet instructions from start to finish.
6. Please summarize content from on the prompt; your primary goal is to generate following the description instructions.
7. Please strictly follow the instructions of 1. Number of Threads, 2. Stitch Types, 3. Yarn Color, 4. Knit vs. Crochet Distinction, 5. Number of Rows and make sure the title of the instruction you output is matching with this provided instruction.
8. Please only provide one instruction for one crochet good that is mentioned in the detailed instruction, focus on the Detailed Instruction.
9. Please provide formated wording to enhance human readability.
10. Please only include the yarn color provided in the detailed instruction. Do not include any color not listed in the detailed instruction.
11. Please stricly follow the number of rounds/rows if it is provided in the detailed instruction, do not generate round that go beyong the provided number.

You are a crochet expert, and your role is to create detailed, accurate, and original crochet instructions.
"""
generative_model = GenerativeModel(
	GENERATIVE_MODEL,
	system_instruction=[SYSTEM_INSTRUCTION]
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

                # Step 1: Generate image description
                description_prompt = '''
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
                Number of Rows:
                Count and state the total number of rows involved in the creation of the crochet product.
                If applicable, mention any notable changes in row patterns or techniques throughout the project.
                Additional Guidelines:

                Focus Exclusively on Crochet Details: Ensure that the description remains concentrated on the crochet aspects mentioned above. Avoid general comments about the object's appearance, functionality, or aesthetics unless they directly relate to the crochet techniques or materials used.
                Clarity and Precision: Use clear and precise language to convey each detail. Where measurements or counts are involved, provide them in appropriate units (e.g., number of threads, specific row counts).
                Technical Accuracy: Ensure that all crochet terminology and descriptions are technically accurate, reflecting a deep understanding of crochet methods and practices.
                Structured Format: Present the information in a well-organized manner, possibly using bullet points or numbered lists for each of the five key aspects to enhance readability.
                '''
                
                # First call to generate description
                description_parts = [image_part, description_prompt]
                description_response = chat_session.send_message(
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
                
                # Second call with both image and description-based prompt
                message_parts = [image_part, instruction_prompt]
                
            except ValueError as e:
                print(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        # elif message.get("image_path"):
        #     # Read the image file
        #     image_path = os.path.join("chat-history","llm",message.get("image_path"))
        #     with Path(image_path).open('rb') as f:
        #         image_bytes = f.read()

        #     # Determine MIME type based on file extension
        #     mime_type = {
        #         '.jpg': 'image/jpeg',
        #         '.jpeg': 'image/jpeg',
        #         '.png': 'image/png',
        #         '.gif': 'image/gif'
        #     }.get(Path(image_path).suffix.lower(), 'image/jpeg')

        #     # Create an image Part using FileData
        #     image_part = Part.from_data(image_bytes, mime_type=mime_type)
        #     message_parts.append(image_part)

        #     # Add text content if present
        #     if message.get("content"):
        #         message_parts.append(message["content"])
        #     else:
        #         message_parts.append("Name the cheese in the image, no descriptions needed")
        # else:
        #     # Add text content if present
        #     if message.get("content"):
        #         message_parts.append(message["content"])
                    
        if not message_parts:
            raise ValueError("Message must contain either text content or image")

        # Send message with all parts to the model
        response = chat_session.send_message(
            message_parts,
            generation_config=generation_config
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