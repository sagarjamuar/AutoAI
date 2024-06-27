# import os
# import torch
# import numpy as np
# import psycopg2
# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Load models and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# bge_model_name = "BAAI/bge-large-en-v1.5"
# api_key = os.getenv("HUGGING_FACE_API_KEY")
#
# # Load Llama model and tokenizer
# tokenizer_llama = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
# model_llama = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key, torch_dtype=torch.float16, device_map="auto")
#
# # Load BGE model and tokenizer for embeddings
# tokenizer_bge = AutoTokenizer.from_pretrained(bge_model_name)
# model_bge = AutoModel.from_pretrained(bge_model_name)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_llama.to(device)
# model_bge.to(device)
#
# # Database connection
# conn = psycopg2.connect(
#     dbname="embeddings_db",
#     user="postgres",
#     password="postgres",
#     host="localhost"
# )
# cur = conn.cursor()
#
# # Function to get embeddings
# def get_embedding(text):
#     inputs = tokenizer_bge(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model_bge(**inputs)
#     embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#     return embedding
#
# # Function to fetch documents from the database
# def fetch_documents():
#     cur.execute("SELECT text, source, embedding FROM documents")
#     rows = cur.fetchall()
#     documents = []
#     for row in rows:
#         text, source, embedding_str = row
#         embedding = np.fromstring(embedding_str.strip('[]'), sep=',')
#         documents.append({"text": text, "source": source, "embedding": embedding})
#     return documents
#
# # Function to find the most similar documents
# def find_similar_documents(query_embedding, documents, top_k=3):
#     embeddings = np.array([doc['embedding'] for doc in documents])
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     top_indices = similarities.argsort()[-top_k:][::-1]
#     return [documents[i] for i in top_indices]
#
# # Flask API setup
# app = Flask(__name__)
#
# @app.route('/query', methods=['POST'])
# def query():
#     user_query = request.json.get('query')
#     query_embedding = get_embedding(user_query)
#
#     documents = fetch_documents()
#     similar_documents = find_similar_documents(query_embedding, documents)
#
#     context = "\n".join([doc['text'] for doc in similar_documents])
#     inputs = tokenizer_llama.encode(context + "\n\n" + user_query, return_tensors="pt").to(device)
#
#     # Ensure input length is within model's limit
#     max_input_length = model_llama.config.max_position_embeddings
#     if inputs.shape[-1] > max_input_length:
#         inputs = inputs[:, -max_input_length:]
#
#     outputs = model_llama.generate(inputs, max_new_tokens=512, num_return_sequences=1)
#
#     response = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({'response': response})
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)






# import os
# import torch
# import numpy as np
# import psycopg2
# import requests
# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Load models and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# api_key = os.getenv("HUGGING_FACE_API_KEY")
# hf_api_key = "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"  # Replace with your Hugging Face API key
# API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
# HEADERS = {"Authorization": f"Bearer {hf_api_key}"}
#
# # Load Llama model and tokenizer
# tokenizer_llama = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
# model_llama = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key, torch_dtype=torch.float16, device_map="auto")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_llama.to(device)
#
# # Database connection
# conn = psycopg2.connect(
#     dbname="embeddings_db",
#     user="postgres",
#     password="postgres",
#     host="localhost"
# )
# cur = conn.cursor()
#
# # Function to get embeddings from Hugging Face API
# def query_hf_api(payload):
#     response = requests.post(API_URL, headers=HEADERS, json=payload)
#     return response.json()
#
# def get_embedding(sentence):
#     payload = {"inputs": sentence}
#     response = query_hf_api(payload)
#     # Assuming the response is a list of embeddings
#     if not isinstance(response, list) or not response:
#         # Handle error if response is not a list or is empty
#         raise ValueError("Invalid response from Hugging Face API")
#     embedding = response  # Assuming the first element is the embedding
#     return embedding
#
# # Function to fetch documents from the database
# def fetch_documents():
#     cur.execute("SELECT text, source, embedding FROM documents")
#     rows = cur.fetchall()
#     documents = []
#     for row in rows:
#         text, source, embedding_str = row
#         embedding = np.fromstring(embedding_str.strip('[]'), sep=',')
#         documents.append({"text": text, "source": source, "embedding": embedding})
#     return documents
#
# # Function to find the most similar documents
# def find_similar_documents(query_embedding, documents, top_k=3):
#     embeddings = np.array([doc['embedding'] for doc in documents])
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     top_indices = similarities.argsort()[-top_k:][::-1]
#     return [documents[i] for i in top_indices]
#
# # Flask API setup
# app = Flask(__name__)
#
# @app.route('/query', methods=['POST'])
# def query():
#     user_query = request.json.get('query')
#     query_embedding = get_embedding(user_query)
#
#     documents = fetch_documents()
#     similar_documents = find_similar_documents(query_embedding, documents)
#
#     context = "\n".join([doc['text'] for doc in similar_documents])
#     inputs = tokenizer_llama.encode(context + "\n\n" + user_query, return_tensors="pt").to(device)
#
#     # Ensure input length is within model's limit
#     max_input_length = model_llama.config.max_position_embeddings
#     if inputs.shape[-1] > max_input_length:
#         inputs = inputs[:, -max_input_length:]
#
#     outputs = model_llama.generate(inputs, max_new_tokens=512, num_return_sequences=1)
#
#     response = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({'response': response})
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

### seperator

# import os
# import torch
# import numpy as np
# import psycopg2
# import requests
# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Load models and tokenizer
# api_key = os.getenv("HUGGING_FACE_API_KEY")
# hf_api_key = "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"  # Replace with your Hugging Face API key
# API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
# HEADERS = {"Authorization": f"Bearer {hf_api_key}"}
#
# # Database connection
# conn = psycopg2.connect(
#     dbname="embeddings_db",
#     user="postgres",
#     password="postgres",
#     host="localhost"
# )
# cur = conn.cursor()
#
# # Function to get embeddings from Hugging Face API
# def query_hf_api(payload):
#     response = requests.post(API_URL, headers=HEADERS, json=payload)
#     return response.json()
#
# def get_embedding(sentence):
#     payload = {"inputs": sentence}
#     response = query_hf_api(payload)
#     if not isinstance(response, list) or not response:
#         raise ValueError("Invalid response from Hugging Face API")
#     embedding = response  # Assuming the first element is the embedding
#     return embedding
#
# # Function to fetch documents from the database
# def fetch_documents():
#     cur.execute("SELECT text, source, embedding FROM documents")
#     rows = cur.fetchall()
#     documents = []
#     for row in rows:
#         text, source, embedding_str = row
#         embedding = np.fromstring(embedding_str.strip('[]'), sep=',')
#         documents.append({"text": text, "source": source, "embedding": embedding})
#     return documents
#
# # Function to find the most similar documents
# def find_similar_documents(query_embedding, documents, top_k=3):
#     embeddings = np.array([doc['embedding'] for doc in documents])
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     top_indices = similarities.argsort()[-top_k:][::-1]
#     return [documents[i] for i in top_indices]
#
# # Flask API setup
# app = Flask(__name__)
#
# @app.route('/query', methods=['POST'])
# def query():
#     user_query = request.json.get('query')
#     query_embedding = get_embedding(user_query)
#
#     documents = fetch_documents()
#     similar_documents = find_similar_documents(query_embedding, documents)
#
#     # Print the text and sources of similar documents
#     for i, doc in enumerate(similar_documents):
#         print(f"Document {i+1} Text:", doc['text'])
#         print(f"Document {i+1} Source:", doc['source'])
#
#     return jsonify({'status': 'success'})
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


#seperator

# import os
# import numpy as np
# import psycopg2
# import requests
# from flask import Flask, request, jsonify
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Load models and tokenizer
# api_key = os.getenv("HUGGING_FACE_API_KEY")
# hf_api_key = "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"  # Replace with your Hugging Face API key
# API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
# HEADERS = {"Authorization": f"Bearer {hf_api_key}"}
#
# # Database connection
# conn = psycopg2.connect(
#     dbname="embeddings_db",
#     user="postgres",
#     password="postgres",
#     host="localhost"
# )
# cur = conn.cursor()
#
# # Function to get embeddings from Hugging Face API
# def query_hf_api(payload):
#     response = requests.post(API_URL, headers=HEADERS, json=payload)
#     return response.json()
#
# def get_embedding(sentence):
#     payload = {"inputs": sentence}
#     response = query_hf_api(payload)
#     if not isinstance(response, list) or not response:
#         raise ValueError("Invalid response from Hugging Face API")
#     embedding = response  # Assuming the first element is the embedding
#     return embedding
#
# # Function to fetch documents from the database
# count=0
# def fetch_documents():
#     cur.execute("SELECT text, source, embedding FROM documents")
#     rows = cur.fetchall()
#     documents = []
#     for row in rows:
#         text, source, embedding_str = row
#         embedding = eval(embedding_str)
#         # embedding = np.fromstring(embedding_str.strip('[]'), sep=',')
#         # print(eval_embedding==embedding)
#
#         documents.append({"text": text, "source": source, "embedding": embedding})
#
#     return documents
#
# # Function to find the most similar documents
# def find_similar_documents(query_embedding, documents, top_k=3):
#     embeddings = np.array([doc['embedding'] for doc in documents])
#
#     # embeddings = np.array([documents['embedding']])
#     # print(embeddings[0])
#     # print(len(embeddings))
#     query_embedding = np.array(query_embedding)
#     similarities = cosine_similarity([query_embedding], embeddings)[0]
#     # print(similarities)
#     top_indices = similarities.argsort()[-top_k:][::-1]
#     return [documents[i] for i in top_indices]
#
# # Flask API setup
# app = Flask(__name__)
#
# @app.route('/query', methods=['POST'])
# def query():
#     user_query = request.json.get('query')
#     query_embedding = get_embedding(user_query)
#
#     documents = fetch_documents()
#     similar_documents = find_similar_documents(query_embedding, documents)
#
#     # Print the text and sources of similar documents
#     # print(len(query_embedding))
#     for i, doc in enumerate(similar_documents):
#         # print(f"Document {i+1} Text:", doc['text'])
#         print(f"Document {i+1} Source:", doc['source'])
#         # print(f"Document {i+1} Text:", len(doc['embedding']))
#
#
#     # return jsonify({'status': 'success', 'similar_documents': [{'text': doc['text'], 'source': doc['source']} for doc in similar_documents]})
#     return jsonify({'status': 'success', 'similar_documents': [{'source': doc['source']} for doc in similar_documents]})
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


import os
import numpy as np
import psycopg2
import requests
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load models and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
api_key = os.getenv("HUGGING_FACE_API_KEY")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key, torch_dtype=torch.float16, device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

hf_api_key = "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"
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
HEADERS = {"Authorization": f"Bearer {hf_api_key}"}

# Database connection
conn = psycopg2.connect(
    dbname="embeddings_db",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

# Function to get embeddings from Hugging Face API
def query_hf_api(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

def get_embedding(sentence):
    payload = {"inputs": sentence}
    response = query_hf_api(payload)
    if not isinstance(response, list) or not response:
        raise ValueError("Invalid response from Hugging Face API")
    embedding = response  # Assuming the first element is the embedding
    return embedding

# Function to fetch documents from the database
def fetch_documents():
    cur.execute("SELECT text, source, embedding FROM documents")
    rows = cur.fetchall()
    documents = []
    for row in rows:
        text, source, embedding_str = row
        embedding = eval(embedding_str)
        documents.append({"text": text, "source": source, "embedding": embedding})
    return documents

# Function to find the most similar documents
def find_similar_documents(query_embedding, documents, top_k=3):
    embeddings = np.array([doc['embedding'] for doc in documents])
    query_embedding = np.array(query_embedding)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Function to generate response using LLaMA model
def generate_response(query, context_texts):
    input_text = f"Question: {query}\n\nContext:\n{'\n'.join(context_texts)}\n\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'], max_length=512, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Answer:")[1].strip() if "Answer:" in response else response
    return answer


# Flask API setup
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    query_embedding = get_embedding(user_query)
    documents = fetch_documents()
    similar_documents = find_similar_documents(query_embedding, documents)

    context_texts = [doc['text'] for doc in similar_documents]
    response_text = generate_response(user_query, context_texts)

    return jsonify({'status': 'success', 'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
