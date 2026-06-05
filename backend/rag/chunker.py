import json
import os
import re
from pathlib import Path

INPUT_DIR   = "data/processed"
OUTPUT_FILE = "data/chunks.json"
CHUNK_SIZE  = 400
OVERLAP     = 80

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, source):
    words   = text.split()
    chunks  = []
    start   = 0
    idx     = 0
    while start < len(words):
        end         = start + CHUNK_SIZE
        chunk_words = words[start:end]
        if len(chunk_words) >= 50 or idx == 0:
            chunks.append({
                "id":          f"{source}__chunk_{idx}",
                "text":        " ".join(chunk_words),
                "source":      source,
                "chunk_index": idx,
                "word_count":  len(chunk_words),
            })
            idx += 1
        start += CHUNK_SIZE - OVERLAP
    return chunks

def main():
    files = list(Path(INPUT_DIR).glob("*.txt"))
    if not files:
        print(f"[Chunker] No files found in {INPUT_DIR}")
        return
    all_chunks = []
    for path in files:
        text   = clean_text(path.read_text(encoding="utf-8"))
        source = path.stem
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"[Chunker] {path.name} → {len(chunks)} chunks")
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"[Chunker] Done. {len(all_chunks)} total chunks → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()