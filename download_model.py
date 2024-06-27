from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from dotenv import load_dotenv

load_dotenv()

model_name = "meta-llama/Llama-2-7b-chat-hf"
api_key = os.getenv("HUGGING_FACE_API_KEY")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key, torch_dtype=torch.float16, device_map="auto")
