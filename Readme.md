# Brightwheel AI Triage System

AI-powered message triage for Brightwheel's Onboarding team. Classifies, prioritizes, routes, and drafts replies for inbound school messages.

**Live demo:** https://smiti24-brightwheel.hf.space

## Stack
- **AI:** Groq (Llama 3.3 70B) + rule-based safety guardrails
- **Backend:** Python + Flask
- **Deployment:** Docker on Hugging Face Spaces

## Run Locally
```bash
git clone https://github.com/YOURUSERNAME/brightwheel-triage
cd brightwheel-triage
pip install -r requirements.txt
# Add GROQ_API_KEY to .env
python app.py
```

## Environment Variables
```
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.3-70b-versatile
USE_AI=true
```

## How It Works
Each message is scrubbed for PII, sent to Groq for classification, then post-processed by deterministic rules that override AI on critical routing decisions (privacy breaches, wrong-team emails, launch blockers). If the API fails, the system falls back to rule-based triage automatically.
