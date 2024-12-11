import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import base64
<<<<<<< HEAD
=======
<<<<<<< HEAD
# import io
=======
>>>>>>> main
>>>>>>> main
from PIL import Image
from pathlib import Path
import traceback
import chromadb
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, ChatSession, Part
<<<<<<< HEAD
from api.utils.llm_image_utils import image_to_vector, image_to_vector_from_bytes  
=======
<<<<<<< HEAD
from api.utils.llm_image_utils import image_to_vector, image_to_vector_from_bytes  # Import the functions
import numpy as np
import uuid
=======
from api.utils.llm_image_utils import image_to_vector, image_to_vector_from_bytes  
>>>>>>> main
>>>>>>> main

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-1.5-flash-002"
CHROMADB_HOST = os.environ["CHROMADB_HOST"]
CHROMADB_PORT = os.environ["CHROMADB_PORT"]
<<<<<<< HEAD
MODEL_ENDPOINT = "projects/376381333238/locations/us-central1/endpoints/3614500440290361344"

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 3000,  # Reduce output length to avoid exceeding model constraints
    "temperature": 0.5,  # Increase randomness
    "top_p": 0.9,  # Broader token sampling
=======

# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 3000,  # Maximum number of tokens for output
    "temperature": 0.1,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
>>>>>>> main
}

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
<<<<<<< HEAD
You are a highly skilled AI assistant specialized in creating crochet patterns. 
Your task is to generate a detailed crochet pattern for the product shown in the image. 
However, you must prioritize user preferences from their input when they differ from the image. 

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
"""

=======
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
>>>>>>> main
generative_model = GenerativeModel(
	GENERATIVE_MODEL,
	system_instruction=[SYSTEM_INSTRUCTION]
)
<<<<<<< HEAD

=======
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#python
>>>>>>> main
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

# Initialize chat sessions
chat_sessions: Dict[str, ChatSession] = {}

# Connect to chroma DB
client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
collection_name = "semantic-text-image-collection"

# Get the collection
collection = client.get_collection(name=collection_name)

def re_rank_results(text_results, image_results, text_weight=0.6, image_weight=0.4):
    """
    Re-rank results based on weighted scores from both text and image searches.

    Args:
    - text_results (dict): The results from the text query.
    - image_results (dict): The results from the image query.
    - text_weight (float): The weight to apply to text scores (default 0.6).
    - image_weight (float): The weight to apply to image scores (default 0.4).

    Returns:
    - ranked_results (list): List of documents sorted by the combined weighted score.
    """
    result_scores = {}

    # Combine scores from text results
    for idx, doc_id in enumerate(text_results["ids"][0]):  # Access list inside "ids"
        score = text_results["distances"][0][idx] * text_weight  # Access corresponding score
        if doc_id not in result_scores:
            result_scores[doc_id] = score
        else:
            result_scores[doc_id] += score  # If doc already exists, sum up the scores

    # Combine scores from image results
    for idx, doc_id in enumerate(image_results["ids"][0]):  # Access list inside "ids"
        score = image_results["distances"][0][idx] * image_weight  # Access corresponding score
        if doc_id not in result_scores:
            result_scores[doc_id] = score
        else:
            result_scores[doc_id] += score  # If doc already exists, sum up the scores

    # Sort the documents based on the combined score (lower distances mean better matches)
    sorted_results = sorted(result_scores.items(), key=lambda item: item[1])

    # Convert the sorted results into a list of dictionaries for better readability
    ranked_results = [{"id": doc_id, "score": score} for doc_id, score in sorted_results]

    return ranked_results

def generate_query_embedding(query):
	query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
	kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
	embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
	return embeddings[0].values

def create_chat_session() -> ChatSession:
    """Create a new chat session with the model"""
    return generative_model.start_chat()

def generate_chat_response(chat_session: ChatSession, message: Dict) -> str:
    """
    Generate a response using the chat session to maintain history.
    Handles both text and image inputs.
    
    Args:
        chat_session: The Vertex AI chat session
        message: Dict containing 'content' (text) and optionally 'image' (base64 string or image path)
    
    Returns:
        str: The model's response
    """
    try:
        # Initialize parts list for the message
        message_parts = []
        
        # Process image if present
        if message.get("image"):
            try:
                # Extract the actual base64 data and mime type
                base64_string = message.get("image")
                if ',' in base64_string:
                    header, base64_data = base64_string.split(',', 1)
                    mime_type = header.split(':')[1].split(';')[0]
                else:
                    base64_data = base64_string
                    mime_type = 'image/jpeg'  # default to JPEG if no header
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_data)
<<<<<<< HEAD
                image_part = Part.from_data(image_bytes, mime_type=mime_type)
=======
>>>>>>> main
                
                # Convert the image bytes to a vector
                image_vector = image_to_vector_from_bytes(image_bytes)

                # Add the image vector to the message
                message["image_embedding"] = image_vector.tolist() 

            except ValueError as e:
                print(f"Error processing image: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Image processing failed: {str(e)}"
                )

        # Process image path if present
        if message.get("image_path"):
            image_path = message["image_path"]
            # Convert the image to a vector
            image_vector = image_to_vector(image_path)

            # Add the image vector to the message
            message["image_embedding"] = image_vector.tolist()  

        # Add text content if present
        if message.get("content"):
            # Create embeddings for the message content
            query_embedding = generate_query_embedding(message["content"])
            
            # Create a dummy image embedding if not provided
            dummy_image_embedding = [0.0] * 1024

            # Retrieve the image embedding from the message
            image_embedding = message.get("image_embedding", dummy_image_embedding) 

            # Concatenate the query embedding and image embedding
            combined_embedding = query_embedding + image_embedding

            # Perform the text query with the combined embedding
            combined_results = collection.query(
                query_embeddings=[combined_embedding],
                n_results=5
            )

            # Perform the image query with only query embedding
            text_results = collection.query(
                query_embeddings=[query_embedding+dummy_image_embedding],
                n_results=5
            )

            # Re-rank the results based on both text and image queries
            ranked_results = re_rank_results(combined_results, text_results, text_weight=0.6, image_weight=0.4)

            # Extract document IDs from ranked results
            result_ids = [result['id'] for result in ranked_results]
            retrieved_data = collection.get(ids=result_ids, include=['documents', 'embeddings'])
            embedded_texts = retrieved_data['documents']
            combined_text_chunks = ' '.join(embedded_texts)

            # Extract the top-ranked document
            if ranked_results:
                top_result = ranked_results[0]['id']
                INPUT_PROMPT = f"""
                {message["content"]}
                {combined_text_chunks}
                """
                message_parts.append(INPUT_PROMPT)
            else:
                message_parts.append("No relevant results found.")

        if not message_parts:
            raise ValueError("Message must contain either text content or image")

<<<<<<< HEAD
        print(f"Message parts: {message["content"]}")
        model_input = [image_part] + message_parts if image_part else message_parts

        # Send message with all parts to the model
        response = chat_session.send_message(
            model_input,
=======
        # Send message with all parts to the model
        response = chat_session.send_message(
            message_parts,
>>>>>>> main
            generation_config=generation_config
        )
        
        print(f"Response: {response.text}")
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
    
    return new_session