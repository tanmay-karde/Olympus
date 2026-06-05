from pathlib import Path


def chunk_text(text, chunk_size=1000):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        # Skip empty paragraphs
        if not para.strip():
            continue

        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Test code
if __name__ == "__main__":

    text = Path(
        "data/processed/PercyJackson.txt"
    ).read_text(
        encoding="utf-8"
    )

    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- CHUNK {i+1} ---")
        print(chunk[:500])