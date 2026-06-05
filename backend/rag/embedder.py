from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "Poseidon is the god of the sea."

embedding = model.encode(sentence)

print(type(embedding))
print(len(embedding))
print(embedding[:10])