import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_ANALYST = os.getenv("MODEL_ANALYST", "gemini-2.5-flash")
MODEL_EDITOR = os.getenv("MODEL_EDITOR", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
