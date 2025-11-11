import os
import base64
import io
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import pkg_resources

print("=== PACKAGE VERSIONS ===")
for name in ["fastapi", "pydantic", "starlette"]:
    try:
        print(name, pkg_resources.get_distribution(name).version)
    except Exception:
        print(name, "NOT FOUND")
print("=========================")

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

# --- PDF PRICE LIST MANAGEMENT ---
from fastapi import UploadFile, File, Form
import fitz  # PyMuPDF

UPLOAD_DIR = "/tmp/pricelists"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PRICELISTS = {}  # {site_url: text_from_pdf}

@app.post("/upload-pricelist")
async def upload_pricelist(file: UploadFile = File(...), site_url: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        PRICELISTS[site_url] = text
    except Exception as e:
        return {"error": str(e)}

    return {"status": "ok", "filename": file.filename, "text_length": len(text)}

@app.post("/delete-pricelist")
async def delete_pricelist(site_url: str = Form(...)):
    PRICELISTS.pop(site_url, None)
    return {"status": "deleted"}


# --- CHAT ENDPOINT ---
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

    # 1️⃣ Získaj dáta z payloadu
    site_text = (payload.site_text or "")[:12000]
    wp_knowledge = (payload.wp_knowledge or "")[:30000]
    site_url = payload.site_url or ""
    language = payload.language or "sk"

    # 2️⃣ Načítaj cenník z pamäte (ak bol nahratý)
    pricelist_text = PRICELISTS.get(site_url, "")

    # 3️⃣ Ak prišiel cenník aj priamo v base64, dekóduj ho
    if not pricelist_text and payload.pricelist:
        try:
            import fitz  # PyMuPDF
            import base64
            import io

            raw = base64.b64decode(payload.pricelist.base64)
            pdf = fitz.open(stream=io.BytesIO(raw), filetype="pdf")
            pricelist_text = ""
            for page in pdf:
                pricelist_text += page.get_text()
            PRICELISTS[site_url] = pricelist_text
        except Exception as e:
            print(f"[PDF decode error] {e}")

    # 4️⃣ Poskladaj kontext pre OpenAI
    context = f"""
[CONTEXT]
SITE_URL: {site_url}
PAGE_TEXT: {site_text}
WP_KNOWLEDGE: {wp_knowledge}
PRICELIST_TEXT: {pricelist_text[:15000]}  # max 15k znakov
""".strip()

system_prompt = (
    "You are TEKBOTIK, a friendly Slovak AI assistant for websites. "
    "Answer only from the provided context. "
    "If you cannot find the answer, reply exactly: "
    "'Prepáč, s týmto si nie som istý, prosím zanechaj email, pošlem ti odpoveď, hneď ako ju preverím'. "
    "Prefer Slovak language when language=sk."
)

user_msg = f"Language: {language}\nQuestion: {payload.question}"

    try:
        import openai
        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )

        reply = completion["choices"][0]["message"]["content"].strip()
        return {"reply": reply}

    except Exception as e:
        return JSONResponse({"reply": f"Server error: {e}"}, status_code=500)
