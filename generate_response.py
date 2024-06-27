import requests
import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Hugging Face and Neo4j Config
API_URL = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
HEADERS = {"Authorization": "Bearer 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"}
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "sagar@sundeus"

# Initialize Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Flask app
app = Flask(__name__)

# Initialize LLM
model_name = "meta-llama/Llama-2-7b-chat-hf"
api_key = os.getenv("HUGGING_FACE_API_KEY")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key, torch_dtype=torch.float16,
                                             device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def query_embedding(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()


def get_embedding(sentence):
    payload = {"inputs": sentence}
    response = query_embedding(payload)
    return np.array(response)


def fetch_embeddings_from_neo4j():
    with driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN d.name AS name, d.embedding AS embedding")
        documents = []
        for record in result:
            name = record["name"]
            embedding = record["embedding"]
            # Convert embedding to numpy array and ensure it is in the correct format
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            elif isinstance(embedding, dict):
                embedding = np.array(list(embedding.values()))
            documents.append((name, embedding))
    return documents


def get_relevant_documents(query_embedding, top_k=1):
    documents = fetch_embeddings_from_neo4j()
    similarities = [(name, cosine_similarity([query_embedding], [embedding])[0][0])
                    for name, embedding in documents if embedding is not None and embedding.size > 0]
    top_documents = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    return [doc[0] for doc in top_documents]


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_query = data['prompt']
    query_emb = get_embedding(user_query)
    relevant_docs = get_relevant_documents(query_emb)

    # Create a simple context using document names
    context = "Relevant documents: " + ", ".join(relevant_docs)

    # Combine user query and context
    prompt = f"User query: {user_query}\n\nContext: {context}\n\nResponse:"

    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response.strip()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# from neo4j import GraphDatabase
# import numpy as np
#
# # Neo4j Config
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "sagar@sundeus"
#
# # Initialize Neo4j
# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
#
#
# def fetch_documents_from_neo4j():
#     with driver.session() as session:
#         result = session.run("MATCH (d:Document) RETURN d.name AS name, d.embedding AS embedding, d.text AS text")
#         documents = [(record["name"], np.array(record["embedding"]), record["text"]) for record in result]
#
#     return documents
#
#
# if __name__ == "__main__":
#     documents = fetch_documents_from_neo4j()
#     for name, embedding, text in documents:
#         print(f"Name: {name}")
#         print(f"Embedding: {embedding}")
#         print(f"Text: {text}")
#         print("-------------------------")
#
