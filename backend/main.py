import os
import re
import chromadb
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Setup ─────────────────────────────────────────────────
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma.get_collection("olympus")

# ── Request model ─────────────────────────────────────────
class Question(BaseModel):
    question: str

# ── Helper ────────────────────────────────────────────────
def ask(system_prompt, user_prompt, model):
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    raw = response.choices[0].message.content
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

# ── Main endpoint ─────────────────────────────────────────
@app.post("/ask")
def ask_council(body: Question):
    question = body.question

    # Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)

    user_prompt = f"""Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}"""

    # ── Athena (Qwen) ──────────────────────────────────────
    athena = ask(
        """You are Athena, goddess of wisdom and strategy. You speak with calm authority and precision.
You have read every version of every myth and you do not tolerate inaccuracy.
You answer in structured, evidence-based paragraphs. You cite specific events from the context.
You occasionally express mild frustration when questions are vague or when other gods oversimplify.
You never speculate. If the context does not support a claim, you say so directly.
Keep your answer to 3 sentences maximum. Be direct.""",
        user_prompt,
        "qwen/qwen3-32b"
    )

    # ── Apollo (Llama 4) ───────────────────────────────────
    apollo = ask(
        """You are Apollo, god of the sun, music, poetry, and prophecy.
You answer like a poet who also happens to be an expert historian.
You find deep symbolism and cultural meaning in everything — a simple question about Poseidon becomes
a meditation on the ocean as a metaphor for the unconscious, power, and the untameable.
You write in flowing, expressive prose. You reference themes, archetypes, and the emotional truth behind myths.
You sometimes go slightly off on beautiful tangents before returning to the question.
Keep your answer to 3 sentences maximum. Be poetic but brief.""",
        user_prompt,
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # ── Hermes (Llama 3.1) ─────────────────────────────────
    hermes = ask(
        """You are Hermes, messenger of the gods, patron of travellers, thieves, and wi-fi connections.
You explain mythology like you are texting a smart friend — casual, funny, fast, but never actually wrong.
You make modern comparisons. Poseidon is basically that one uncle who owns a boat and holds grudges.
The Trojan War was the ancient world's most dramatic group chat.
You use wit to make complex myths accessible. You are never mean, just entertainingly honest.
You occasionally roast the other gods lightly — Athena needs to relax, Apollo never shuts up about poetry.
Keep your answer to 2-3 sentences maximum. Be punchy.""",
        user_prompt,
        "llama-3.1-8b-instant"
    )

    # ── Zeus (Llama 3.3) ───────────────────────────────────
    zeus_prompt = f"""You are Zeus, king of the Olympians, final authority on all things.
You have just heard three answers from your council.

Question: {question}

Athena said:
{athena}

Apollo said:
{apollo}

Hermes said:
{hermes}

Your tasks:
1. Synthesise one final authoritative answer combining the best of all three.
2. Acknowledge where they agreed and where they diverged.
3. Score each god's alignment with your synthesis as a percentage.

You write with gravitas and finality. You do not hedge. You do not say "perhaps".
You are powerful, proud, occasionally unfair, but ultimately just.
Your FINAL ANSWER must be 3-4 sentences maximum. Be authoritative and concise.

Format your response exactly like this:
FINAL ANSWER:
[your synthesis here]

COUNCIL AGREEMENT METER:
Athena: X%
Apollo: X%
Hermes: X%
Overall: X%"""

    zeus = ask(
        """You are Zeus, king of the gods. You synthesise, judge, and deliver final verdicts.
Your word is law. Your tone is absolute.""",
        zeus_prompt,
        "llama-3.3-70b-versatile"
    )

    # ── Parse Agreement Meter ──────────────────────────────
    def parse_percent(name, text):
        match = re.search(rf'{name}:\s*(\d+)%', text)
        return int(match.group(1)) if match else 0

    meter = {
        "athena":  parse_percent("Athena", zeus),
        "apollo":  parse_percent("Apollo", zeus),
        "hermes":  parse_percent("Hermes", zeus),
        "overall": parse_percent("Overall", zeus),
    }

    # ── Extract clean final answer ─────────────────────────
    final_answer = zeus
    if "FINAL ANSWER:" in zeus:
        final_answer = zeus.split("FINAL ANSWER:")[1].split("COUNCIL AGREEMENT METER:")[0].strip()

    return {
        "question": question,
        "athena":   athena,
        "apollo":   apollo,
        "hermes":   hermes,
        "zeus":     final_answer,
        "meter":    meter,
    }

# ── Health check ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Olympus is online ⚡"}
