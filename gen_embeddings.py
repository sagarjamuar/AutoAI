import requests
import os
from PyPDF2 import PdfReader
import docx

API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
HEADERS = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def get_embedding(sentence):
    payload = {"inputs": sentence}
    response = query(payload)
    return response

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_files(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(os.path.join(root, filename))
            elif filename.endswith('.docx'):
                text = extract_text_from_docx(os.path.join(root, filename))
            else:
                continue
            embedding = get_embedding(text)
            documents[filename] = embedding

# Process files and get embeddings
documents = {}
for folder in ['/home/ubuntu/Vehicle_Info/INFO', '/home/ubuntu/Vehicle_Info/SPECS']:
    process_files(folder)

# Save embeddings
import pickle
with open('/home/ubuntu/Vehicle_Info/embeddings.pkl', 'wb') as f:
    pickle.dump(documents, f)
