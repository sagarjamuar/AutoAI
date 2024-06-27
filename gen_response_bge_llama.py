import os
import torch
import numpy as np
import psycopg2
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Load models and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
bge_model_name = "BAAI/bge-large-en-v1.5"
api_key = os.getenv("HUGGING_FACE_API_KEY")

# Load Llama model and tokenizer
tokenizer_llama = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
model_llama = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key, torch_dtype=torch.float16, device_map="auto")

# Load BGE model and tokenizer for embeddings
tokenizer_bge = AutoTokenizer.from_pretrained(bge_model_name)
model_bge = AutoModel.from_pretrained(bge_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llama.to(device)
model_bge.to(device)

# Database connection
conn = psycopg2.connect(
    dbname="embeddings_db",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer_bge(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_bge(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Function to fetch documents from the database
def fetch_documents():
    cur.execute("SELECT text, source, embedding FROM documents")
    rows = cur.fetchall()
    documents = []
    for row in rows:
        documents.append({"text": row[0], "source": row[1], "embedding": np.array(row[2])})
    return documents

# Function to find the most similar documents
def find_similar_documents(query_embedding, documents, top_k=3):
    embeddings = np.array([doc['embedding'] for doc in documents])
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Flask API setup
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    query_embedding = get_embedding(user_query)

    documents = fetch_documents()
    similar_documents = find_similar_documents(query_embedding, documents)

    context = "\n".join([doc['text'] for doc in similar_documents])
    inputs = tokenizer_llama.encode(context + "\n\n" + user_query, return_tensors="pt").to(device)
    outputs = model_llama.generate(inputs, max_length=512, num_return_sequences=1)

    response = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
