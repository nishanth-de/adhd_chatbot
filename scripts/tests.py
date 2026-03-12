import os
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
api_key = os.getenv("GEMINI_API_KEY")

print(f"DB URL loaded: {'YES' if db_url else 'NO'}")
print(f"API Key loaded: {'YES' if api_key else 'NO'}")