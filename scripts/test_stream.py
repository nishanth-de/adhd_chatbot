import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # Try the stream method
    response = client.models.generate_content_stream(
        model = "gemini-2.5-flash-lite",
        contents = "Write a 100 word explanation of what ADHD is.",
        config = types.GenerateContentConfig(
            temperature = 0.0,
            max_output_tokens = 100
        )
    )
    
    chunk_count = 0
    for chunk in response:
        if chunk.text:
            chunk_count += 1
            print(chunk.text, end="", flush=True)

    print(f"\n{'-'*40}")
    print(f"Total chunks received: {chunk_count}")
    print("If chunk_count > 1, streaming is working correctly")


except AttributeError as e:
    print(f"Method not found: {e}")
    print("Your SDK version may use a different method name")

except Exception as e:
    print(f"Error: {e}")