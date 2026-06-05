import json
import os
import time
from sentence_transformers import SentenceTransformer

INPUT_FILE  = "data/chunks.json"
OUTPUT_FILE = "data/embedded_chunks.json"
BATCH_SIZE  = 64

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[Embedder] {len(chunks)} chunks loaded")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    t0    = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"[Embedder] Done in {time.time()-t0:.1f}s")
    embedded = []
    for chunk, emb in zip(chunks, embeddings):
        embedded.append({**chunk, "embedding": emb.tolist()})
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded, f, ensure_ascii=False)
    print(f"[Embedder] Saved {len(embedded)} embedded chunks → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()