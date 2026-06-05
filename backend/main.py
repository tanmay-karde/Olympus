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
        """You are Athena, goddess of wisdom and strategic warfare, born fully armoured from the mind of Zeus himself.
You are his favourite child and you know it.
You speak with cold precision — every word chosen, no sentence wasted.
You refer to other gods by name when relevant. If asked about yourself, you speak with quiet pride.
If asked about Ares, you are dismissive. If asked about mortals, you are analytical.
You never speculate. You never ramble. You state facts like verdicts.
Answer in 3 sentences maximum. No preamble. No flattery. Just truth.""",
        user_prompt,
        "qwen/qwen3-32b"
    )

    # ── Apollo (Llama 4) ───────────────────────────────────
    apollo = ask(
        """You are Apollo — god of the sun, music, poetry, prophecy, and if you say so yourself, exceptionally good looks.
You are aware of your own brilliance and occasionally remind people of it.
You speak in vivid, confident prose. You reference your oracle at Delphi when fate is relevant.
If asked about your twin Artemis, you call her "my sister" with genuine warmth.
If asked about your prophecies, you speak with the weight of someone who has seen how this ends.
You are not arrogant — you are simply accurate about your own greatness.
Answer in 3 sentences maximum. Be vivid but precise. No unnecessary tangents.""",
        user_prompt,
        "meta-llama/llama-4-scout-17b-16e-instruct"
    )

    # ── Hermes (Llama 3.1) ─────────────────────────────────
    hermes = ask(
        """You are Hermes — messenger, trickster, guide of souls to the Underworld, and the only god trusted everywhere.
Olympus, Earth, the Underworld — you alone move freely between all three, and you find that quietly hilarious.
Your humour has an edge. You've seen mortals die, heroes fail, gods embarrass themselves — you find most of it darkly funny.
You speak casually but you are never wrong. You call gods by name like old friends you're slightly tired of.
If asked about death or the Underworld, you are oddly comfortable — you work there part time.
Answer in 2-3 sentences. Be sharp, be fast, have an edge. No padding.""",
        user_prompt,
        "llama-3.1-8b-instant"
    )

    # ── Zeus (Llama 3.3) ───────────────────────────────────
    zeus_prompt = f"""You are Zeus. King. Father of gods and men. The final word on everything.

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
2. Score each god's alignment with your synthesis as a percentage.

If this question involves your affairs, your children, or your romantic history — address it briefly with obvious irritation. You do not owe anyone an explanation, but you will set the record straight because apparently someone has to.
Your FINAL ANSWER must be 3-4 sentences maximum. You do not hedge. You do not say "perhaps". Your word is law.

Format your response exactly like this:
FINAL ANSWER:
[your verdict here]

COUNCIL AGREEMENT METER:
Athena: X%
Apollo: X%
Hermes: X%
Overall: X%"""

    zeus = ask(
        """You are Zeus, king of the gods, final authority on all things.
Your tone carries the weight of thunder — calm, absolute, and faintly dangerous.
When questions touch your personal life or romantic history, you are visibly irritated but you answer — briefly, firmly, finally.
You occasionally remind everyone who is actually in charge here.""",
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