import psycopg2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="embeddings_db",
    user="postgres",
    password="postgres",
    host="localhost"
)
cur = conn.cursor()

# Create table with VECTOR type
cur.execute("""
    CREATE TABLE IF NOT EXISTS debug (
        id SERIAL PRIMARY KEY,
        text TEXT,
        source TEXT,
        embedding VECTOR(3)  -- Adjust the dimension as needed
    );
""")
conn.commit()

# Sample vector
sample_vector = np.array([0.1, 0.2, 0.3])

# Insert sample vector into the table
cur.execute(
    "INSERT INTO debug (text, source, embedding) VALUES (%s, %s, %s)",
    ("Sample text", "sample_source", sample_vector.tolist())
)
conn.commit()

# Retrieve the vector
cur.execute("SELECT embedding FROM debug WHERE source = 'sample_source'")
retrieved_vector = eval(cur.fetchone()[0])

# Convert the retrieved vector to numpy array
retrieved_vector = np.array(retrieved_vector)

similarity = cosine_similarity([sample_vector], [retrieved_vector])

print(f"Sample Vector: {sample_vector}")
print(f"Retrieved Vector: {retrieved_vector}")
print(f"Cosine Similarity: {similarity}")

# Clean up
cur.execute("DROP TABLE debug")
conn.commit()

# Close the database connection
cur.close()
conn.close()
