import os
from pathlib import Path

INPUT_DIR  = "data/raw_texts"
OUTPUT_DIR = "data/processed"

def clean_gutenberg(text):
    """Strip Gutenberg header and footer boilerplate."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            text = text[text.find("\n")+1:]
            break
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()

def ingest_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = list(Path(INPUT_DIR).glob("*.txt"))
    if not files:
        print(f"[Ingestor] No .txt files found in {INPUT_DIR}")
        return
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_gutenberg(text)
        out  = Path(OUTPUT_DIR) / path.name
        out.write_text(text, encoding="utf-8")
        print(f"[Ingestor] {path.name} → {len(text):,} chars → {out}")
    print(f"[Ingestor] Done. {len(files)} files processed.")

if __name__ == "__main__":
    ingest_all()