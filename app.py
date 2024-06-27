from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

model_name = "meta-llama/Llama-2-7b-chat-hf"
api_key = os.getenv("HUGGING_FACE_API_KEY")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, token=api_key, torch_dtype=torch.float16, device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
