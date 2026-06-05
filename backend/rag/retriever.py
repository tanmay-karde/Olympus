import chromadb

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection("percy_jackson")

results = collection.query(
    query_texts=["Poseidon"],
    n_results=3
)

print(results)
print(results["documents"])