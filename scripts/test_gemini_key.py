import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_" \
    "KEY not found in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

# Simplest possible test — list available models
print("Testing Gemini API connection...")
try:
    # Make a minimal embedding call to verify the key works
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents="test connection",
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768 # <--- Set to 768 and default is 3072
        )
    )
    print(f"API key is valid")
    print(f"Embedding model responding")
    print(f"Vector dimensions: {len(response.embeddings[0].values)}")

except Exception as e:
    print(f"Connection failed: {e}")
    print("Check your GEMINI_API_KEY in .env")
    sys.exit(1)

