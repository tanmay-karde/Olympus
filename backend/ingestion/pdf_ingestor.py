import fitz

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return text


pdf_path = "data/raw_pdfs/PercyJackson.pdf"

text = extract_pdf_text(pdf_path)

with open("data/processed/PercyJackson.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("PDF processed successfully")
print(f"Characters extracted: {len(text)}")