# scripts/find_stream_method.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("=== SDK Version ===")
import importlib.metadata
try:
    version = importlib.metadata.version("google-genai")
    print(f"google-genai version: {version}")
except Exception:
    print("Could not determine version")

print("\n=== Available methods on client.models ===")
methods = [m for m in dir(client.models) if not m.startswith("_")]
for method in methods:
    print(f"  {method}")

print("\n=== Methods containing 'stream' ===")
stream_methods = [m for m in dir(client.models) if "stream" in m.lower()]
if stream_methods:
    for m in stream_methods:
        print(f"  {m}")
else:
    print("  None found — checking generate_content signature...")
    import inspect
    try:
        sig = inspect.signature(client.models.generate_content)
        print(f"  generate_content params: {list(sig.parameters.keys())}")
    except Exception as e:
        print(f"  Could not inspect: {e}")