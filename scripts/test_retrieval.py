import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.retrieval import hybrid_search
from app.services.llm import generate_answer

def test_full_pipeline(question: str):
    print(f"\n{'='*60}")
    print(f"Q: {question}")

    # Step 1: Retrieve
    chunks = hybrid_search(question, top_n=4)

    if not chunks:
        print("A: I don't have reliable information about that in my "
            "knowledge base. Please consult a healthcare professional.")
        return
    
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"[{i+1}] {c['source_file']} chunk {c['chunk_index']} "
            f"(rrf={c.get('rrf_score', 'N/A')})")
        
    # Step 2: Generate
    print("\nGenerating answer...")
    answer = generate_answer(question, chunks)
    print(f"\nA: {answer}")

    # Step 3: Show sources (grounded citations foundation)
    print(f"\nSources used:")
    seen = set()
    for c in chunks:
        key = f"{c['source_file']} (chunk {c['chunk_index']})"
        if key not in seen:
            print(f"{key}")
            seen.add(key)



if __name__ == "__main__":
    questions = [
        "What is ADHD and how is it diagnosed?",
        "How does executive dysfunction affect everyday tasks?",
        "What is rejection sensitive dysphoria?",
    ]

    for q in questions:
        test_full_pipeline(q)