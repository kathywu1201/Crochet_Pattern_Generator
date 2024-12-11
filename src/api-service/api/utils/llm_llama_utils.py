import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
from PIL import Image
import traceback
from vertexai.generative_models import GenerativeModel, ChatSession, Part
import requests

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
MODAL_API_URL = "https://wwww0203--llama-predict2-llamamodel-predict.modal.run"
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
Number of Rows:
Count and state the total number of crocheted rows involved in the creation of the crochet product. Do not overly count the number of rows.
If applicable, mention any notable changes in row patterns or techniques throughout the project.
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

<<<<<<< HEAD
=======
LLAMMA_INSTRUCTION = """
You are a highly skilled AI assistant specialized in creating precise, step-by-step crochet patterns. Your goal is to generate a crochet pattern based on the given image description. The pattern must be detailed, easy to follow, and structured for both beginners and experts.

Please include the following in your response:

1. **Introduction**:
   - Briefly describe the project in one sentence.

2. **Materials Needed**:
   - Specify the yarn type, weight, recommended colors, crochet hook size, and any other tools required.

3. **Abbreviations** (if necessary):
   - Include common crochet abbreviations used in the pattern (e.g., sc = single crochet, ch = chain).

4. **Pattern Instructions**:
   - Provide step-by-step instructions to recreate the item.
   - Clearly specify the number of stitches, rows, and rounds for each step.
   - Use precise language to describe each action, such as "In Row 1, crochet 8 single stitches."
   - Highlight any special techniques or stitches used.
   - Ensure the instructions are concise and focused on the crochet process.

5. **Finishing Touches**:
   - Include any final steps needed to complete the project, such as weaving in ends or blocking the piece.

Your response should focus solely on the crochet pattern, avoiding any extraneous details or descriptions unrelated to the crafting process.
"""

>>>>>>> main
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
    try:
        message_parts = []
        generated_description = None
        
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
                # Add a text prompt along with the image
                description_parts = [
                    image_part,
                    "Please analyze this crochet item and provide a detailed description of the crochet item."
                ]
                description_response = description_model.start_chat().send_message(
                    description_parts,
                    generation_config=generation_config
                )
                generated_description = description_response.text

                # Prepare data for Modal LLaMA API
                base64_string = message.get("image")
                if ',' in base64_string:
                    _, base64_data = base64_string.split(',', 1)
                else:
                    base64_data = base64_string

                # Create prompt combining description and user message
                prompt = f"""
                Based on this image and the following description:

                {generated_description}

                {message.get('content', '')}

                Please provide detailed crochet instructions for recreating this crocheted item in the image.
                And remember to add the abbreviation for 'next line' when finishing each round of instruction.
                """

                # prompt = f"""
                # Here is the user prompt:

                # {message.get('content')}

                # Please provide detailed crochet instructions for recreating this crocheted item in the image.
                # """
                print(">>>", prompt)
                
                # Modal API call
                response = requests.post(
                    MODAL_API_URL,
                    files={'image': ('image.jpg', base64.b64decode(base64_data), mime_type)},  
                    data={'description': prompt} 
                )
                
                print("--output response---------------------")
                print(response.json()['output'])

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Failed to get response from Modal API"
                    )
                
                return response.json()['output']
                
            except ValueError as e:
                print(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        else:
            # For text-only messages, call Modal API directly
            response = requests.post(
                MODAL_API_URL,
                data={'description': message.get('content', '')}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to get response from Modal API"
                )
            
            return response.json()['output']
            
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