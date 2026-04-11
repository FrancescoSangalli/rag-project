import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM_PROMPT = """Sei un sistema di supporto alle decisioni preciso e affidabile.
Rispondi SOLO basandoti sul contesto fornito tra i tag [Fonte N].
Per ogni affermazione includi la citazione [Fonte N].
Se il contesto non contiene informazioni sufficienti, dichiaralo esplicitamente.
Non inventare mai informazioni."""

def generate_answer(query: str, context_chunks: list) -> dict:
    if not context_chunks:
        return {
            "answer": "Nessun contesto rilevante trovato nei documenti caricati.",
            "sources": [],
            "tokens_used": 0
        }

    context_str = "\n\n".join([
        f"[Fonte {c['rank']}] (file: {c['source']} | pag. {c['page']})\n{c['text']}"
        for c in context_chunks
    ])

    user_message = f"Contesto:\n{context_str}\n\nDomanda: {query}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        max_tokens=1024
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": context_chunks,
        "tokens_used": response.usage.total_tokens
    }