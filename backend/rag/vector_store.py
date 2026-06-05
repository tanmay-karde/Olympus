import chromadb

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    "olympus_test"
)

results = collection.get()

print(results)