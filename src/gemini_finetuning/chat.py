# chat_model.py
import os
from vertexai.generative_models import GenerativeModel
import json

# Configuration for content generation
generation_config = {
    "max_output_tokens": 1000,
    "temperature": 0.75,
    "top_p": 0.95,
}

SYSTEM_INSTRUCTION = """
    You are a highly skilled AI assistant specialized in creating crochet patterns. Your task is to generate a detailed crochet pattern for the product shown in the image. However, you must prioritize user preferences from their input when they differ from the image. For example, if the user specifies a "red crochet product" but the image shows a "blue crochet product," adjust the pattern accordingly (e.g., update the color of the yarn in the "Materials Needed" section).

    Your response must strictly follow the format below, ensuring the number of rounds or steps matches the typical requirements for the product. Do not generate more rounds or steps than necessary to complete the project. Avoid unnecessary line breaks or redundant information to ensure the output is clear, organized, and precise.
v
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


def chat():
    print("Testing the fine-tuned model with a sample prompt...")

    # MODEL_ENDPOINT = "projects/690419079051/locations/us-central1/endpoints/1726797305073369088"  # Replace with your endpoint ID
    MODEL_ENDPOINT = "projects/376381333238/locations/us-central1/endpoints/3614500440290361344"

    # Load the fine-tuned model
    generative_model = GenerativeModel(
        MODEL_ENDPOINT, 
        system_instruction=[SYSTEM_INSTRUCTION]
        )

    # Sample query prompt to generate pattern instructions
    
    # with open('retrieved_data_v2.json', 'r') as file:
    #     data = json.load(file)
    # query = data.get("prompt") 
    # print("Input prompt:", query)
    query = """
    Based on this image and the following description:

                Here's a detailed analysis of the crochet heart, focusing exclusively on the crochet work:

**Number of Threads:**  It's impossible to determine the exact number of threads used without unraveling the piece. However, it appears to be made with a medium-weight yarn, likely a worsted weight or DK weight, based on the visible thickness of the strands.  The number of threads would be dependent on the length of the yarn used and the yarn's ply.

**Stitch Types:** The heart is primarily worked in double crochet (dc) stitches.  The main body of the heart utilizes a simple dc stitch pattern, creating a relatively even texture. The border is a single crochet (sc) stitch, creating a neat and defined edge.  There is no evidence of other stitch types such as half-double crochet (hdc) or treble crochet (tr).

**Yarn Color:** The heart is crocheted using two colors of yarn. The majority of the heart is a light, pastel blue. The border is a creamy off-white or very pale beige. The color transition between the blue and the off-white is clean and precise.

**Knit vs. Crochet Distinction:** The object is definitively crocheted.  The characteristic loops and the visible individual stitches are indicative of crochet, not knitting.  Knitting produces a different type of fabric structure with interlocking loops that run along the length of the piece, whereas crochet stitches are created individually and joined together. The clear definition of each stitch and the way the stitches are joined together are strong indicators of crochet.

**Number of Rows:**  Precise row counting is difficult due to the shape and the image resolution. However, a reasonable estimate would place the number of rows in the blue section of the heart at approximately 15-20 rows, depending on the height of the double crochet stitches. The border appears to be approximately 2-3 rows of single crochet.  There is no significant change in row patterns or techniques observed.




                heart
    """

    # Generate content from the fine-tuned model
    response = generative_model.generate_content(
        [query],  # Input prompt
        generation_config=generation_config,  # Configuration settings
        stream=False,  # Disable streaming
    )
    generated_text = response.text
    output_data = {
        "prompt": generated_text
    }

    # Output the data to a JSON file
    # json_filename = "output_v14.json"
    # with open(json_filename, 'w') as json_file:
    #     json.dump(output_data, json_file, indent=4)
    print("Fine-tuned LLM Response:", generated_text)

if __name__ == "__main__":
    chat()
