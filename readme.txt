# TEKBOTIK CHAT Backend v2

This backend powers the TEKBOTIK CHAT WordPress plugin.

## Endpoints
- /health → check server
- /chat → main chat endpoint

## Run locally
pip install -r requirements.txt
export OPENAI_API_KEY=sk-proj-xxxx
uvicorn server:app --reload --port 8000

Then open http://localhost:8000/health
