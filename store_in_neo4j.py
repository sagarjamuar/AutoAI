from neo4j import GraphDatabase
import pickle

# Connect to Neo4j
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "sagar@sundeus"))

with open('/home/ubuntu/Vehicle_Info/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

def store_embeddings_in_neo4j(embeddings):
    with driver.session() as session:
        for doc_name, embedding in embeddings.items():
            session.run("""
                MERGE (d:Document {name: $name})
                ON CREATE SET d.embedding = $embedding
                ON MATCH SET d.embedding = $embedding
                """, name=doc_name, embedding=embedding)

store_embeddings_in_neo4j(embeddings)
