# Example Testing Script
import requests

url = "https://wwww0203--llama-predict2-llamamodel-predict.modal.run"

# Path to your test image
image_path = "test/images/heart.png"
user_input = "Show me how to make this heart coaster. This is a 3.5-inch-wide product, limit rounds to fewer than 15"

# Prepare the request
files = {'image': open(image_path, 'rb')}  # Optional image
data = {'user_input': user_input}  # Pass the user input

# Send the request
response = requests.post(url, data=data, files=files)
result = response.json().get('output', '')
print(result)




