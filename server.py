import os
import base64
import io
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="TEKBOTIK CHAT Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pricelist(BaseModel):
    name: str
    mime: str
    base64: str

class ChatPayload(BaseModel):
    question: str
    site_url: Optional[str] = None
    site_text: Optional[str] = ""
    wp_knowledge: Optional[str] = ""
    pricelist: Optional[Pricelist] = None
    language: Optional[str] = "sk"

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}

@app.post("/chat")
async def chat(payload: ChatPayload):
    if not OPENAI_API_KEY:
        return JSONResponse({"reply": "Missing OpenAI API key on server."}, status_code=500)

    site_text = (payload.site_text or "")[:16000]
    wp_knowledge = (payload.wp_knowledge or "")[:180000]
    pricelist_text = ""

    if payload.pricelist:
        try:
            raw = base64.b64decode(payload.pricelist.base64)
            pricelist_text = f"[Pricelist loaded: {payload.pricelist.name}, {len(raw)} bytes]"
        except Exception:
            pricelist_text = "[Failed to read pricelist]"

    system_prompt = (
        "You are TEKBOTIK, a helpful AI assistant for websites. "
        "Answer only from the provided context. "
        "If unsure, say you don't know. "
        "Prefer Slovak language when language=sk."
    )

    context = f"""[CONTEXT]
SITE_URL: {payload.site_url}
PAGE_TEXT: {site_text}
WP_KNOWLEDGE: {wp_knowledge}
PRICELIST: {pricelist_text}"""

    user_msg = f"Language: {payload.language}\nQuestion: {payload.question}"

    try:
        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return {"reply": completion["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        return JSONResponse({"reply": f"Server error: {e}"}, status_code=500)
