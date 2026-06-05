import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Connect to ChromaDB
chroma = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma.get_collection("percy_jackson")

# ── Question ──────────────────────────────────────────────
question = "Who is Poseidon?"

# ── Retrieve relevant chunks ──────────────────────────────
results = collection.query(
    query_texts=[question],
    n_results=5
)
chunks = results["documents"][0]
context = "\n\n".join(chunks)

# ── Helper ────────────────────────────────────────────────
def ask(system_prompt, user_prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

user_prompt = f"""Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}"""

# ── Athena (Qwen) ─────────────────────────────────────────
print("\n🦉 ATHENA:")
athena = ask(
    "You are Athena. Answer analytically and factually. Cite evidence from the context. No speculation.",
    user_prompt,
    "qwen/qwen3-32b"
)
print(athena)

# ── Apollo (Mixtral) ──────────────────────────────────────
print("\n☀️ APOLLO:")
apollo = ask(
    "You are Apollo. Answer with cultural depth, symbolism, and thematic meaning. Be scholarly and expressive.",
    user_prompt,
    "meta-llama/llama-4-scout-17b-16e-instruct"
)
print(apollo)

# ── Hermes (Llama) ────────────────────────────────────────
print("\n🪽 HERMES:")
hermes = ask(
    "You are Hermes. Answer conversationally and wittily. Connect to modern retellings like Percy Jackson. Keep it fun.",
    user_prompt,
    "llama-3.1-8b-instant"
)
print(hermes)

# ── Zeus (Llama 3.1 70B) ──────────────────────────────────
print("\n⚡ ZEUS (Final Verdict):")
zeus_prompt = f"""You are Zeus. You have read three answers from your council.

Question: {question}

Athena said:
{athena}

Apollo said:
{apollo}

Hermes said:
{hermes}

Your tasks:
1. Synthesise one final authoritative answer combining the best of all three.
2. Score each god's alignment with your synthesis as a percentage.

Format your response exactly like this:
FINAL ANSWER:
[your synthesis here]

COUNCIL AGREEMENT METER:
Athena: X%
Apollo: X%
Hermes: X%
Overall: X%"""

zeus = ask(
    "You are Zeus, king of the gods. You synthesise and judge.",
    zeus_prompt,
    "llama-3.3-70b-versatile"
)
print(zeus)