from pathlib import Path
import chromadb

# Import your chunking function
from chunker import chunk_text

# Read Percy Jackson text
text = Path("data/processed/PercyJackson.txt").read_text(
    encoding="utf-8"
)

# Create chunks
chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks")

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

# Create a new collection
collection = client.get_or_create_collection(
    name="percy_jackson"
)

# Store chunks
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"]
    )

print("All chunks stored successfully!")