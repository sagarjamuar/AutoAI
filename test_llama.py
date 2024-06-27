from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model_name = "meta-llama/Llama-2-7b-chat-hf"
api_key = os.getenv("HUGGING_FACE_API_KEY")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key, torch_dtype=torch.float16, device_map="auto")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the prompt
prompt = "Hello, how are you?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the response
print("Response:", response)

