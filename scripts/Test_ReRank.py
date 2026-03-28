import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import cohere

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    print("ERROR: COHERE_API_KEY not found in .env")
    sys.exit(1)

cohere = cohere.ClientV2(api_key=api_key)

print("Testing Cohere Rerank API...")

# Minimal rerank test with a fake documents
response = cohere.rerank(
    model="rerank-english-v3.0",
    query="What is executive dysfunction in ADHD?",
    documents=[
        "Executive dysfunction affects planning and task initiation in ADHD.",
        "The weather in Chennai is typically hot and humid.",
        "People with ADHD struggle with working memory and focus.",
        "ADHD diagnosis requires clinical assessment by a professional."
    ],
    top_n=3
)

print(f"Results returned: {len(response.results)}")
print(f"\nRanking test (should rank ADHD content highest):")

for r in response.results:
    print(
        f"Index {r.index} | score={round(r.relevance_score, 4)} | "
        f"{response.results[r.index] if hasattr(response, 'documents') else ''}"
    )

# Verify the weather chunk scored lowest
scores = [(r.index, r.relevance_score) for r in response.results]
print(f"\nAll scores by original index: {scores}")
print("Weather chunk should have the lowest relevance score")