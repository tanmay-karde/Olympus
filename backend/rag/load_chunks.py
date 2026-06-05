import json
import chromadb

INPUT_FILE      = "data/embedded_chunks.json"
COLLECTION_NAME = "olympus"
CHROMA_PATH     = "data/chroma_db"
BATCH_SIZE      = 100

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[Loader] {len(chunks)} embedded chunks loaded")

    client     = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete existing collection so we start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[Loader] Deleted existing collection '{COLLECTION_NAME}'")
    except:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    print(f"[Loader] Created collection '{COLLECTION_NAME}'")

    # Load in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        collection.add(
            ids        = [c["id"] for c in batch],
            documents  = [c["text"] for c in batch],
            embeddings = [c["embedding"] for c in batch],
            metadatas  = [{"source": c["source"]} for c in batch],
        )
        print(f"[Loader] Loaded {min(i+BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"[Loader] Done. Collection '{COLLECTION_NAME}' ready.")

if __name__ == "__main__":
    main()