import logging
from app.services.retrieval import hybrid_search
from app.services.reranker import rerank_chunks, get_overall_confidence
from app.services.citations import build_citations
from app.services.llm import generate_answer

logger = logging.getLogger(__name__)

def run_rag_pipeline(question: str) -> dict:
    """
    Orchestrates the complete RAG pipeline:
    1. Hybrid search (vector + FTS + RRF)
    2. Cohere reranking
    3. Answer generation
    4. Citation building

    This is the single function your chat endpoint calls.
    Returns a structured dict matching ChatResponse schema.

    Arguments:
        question: user's natural language question

    Returns:
        dict with: answer, confidence, sources, retrieved_count
    """
    logger.info(f"RAG pipeline start | question='{question[:80]}'")

    # Stage 1: Hybrid retrieval
    hybrid_chunks = hybrid_search(question, top_n=5)

    # Confidence gate — no relevant chunks found
    if not hybrid_chunks:
        logger.warning(f"No relevant chunks found for: '{question[:60]}'")
        return {
            "answer": (
                "I don't have reliable information about that in my "
                "knowledge base. For accurate ADHD information, please "
                "consult a healthcare professional or visit CHADD.org."
            ),
            "confidence": "out_of_scope",
            "sources": [],
            "retrieved_count": 0
        }    
    
    # Stage 2: Reranking
    reranked_chunks = rerank_chunks(question, hybrid_chunks, top_n=3)

    # Stage 3: Confidence assessment
    confidence = get_overall_confidence(reranked_chunks)

    # Stage 4: Answer generation
    answer = generate_answer(question, reranked_chunks)

    # Stage 5: Build citations
    citations = build_citations(reranked_chunks)

    logger.info(
        f"RAG pipeline complete | "
        f"confidence={confidence} | "
        f"sources={len(citations)}"
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": citations,
        "retrieved_count": len(hybrid_chunks)
    }

if __name__ == "__main__":
    print("=== Complete RAG Pipeline Test ===\n")

    test_questions = [
        "What is rejection sensitive dysphoria and how does it affect relationships?",
        "Why do people with ADHD struggle with time management?",
        "What coping strategies help with ADHD executive dysfunction?",
        "What is the capital of France?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")

        result = run_rag_pipeline(question)

        print(f"\nConfidence: {result['confidence']}")
        print(f"Retrieved: {result['retrieved_count']} chunks")
        print(f"\nAnswer:\n{result['answer']}")

        if result['sources']:
            print(f"\nSources ({len(result['sources'])}):")
            for i, src in enumerate(result['sources'], 1):
                print(
                    f"  [{i}] {src['source_file']} "
                    f"chunk {src['chunk_index']} "
                    f"(relevance={src['relevance_score']}, "
                    f"{src['confidence']})"
                )
                print(f"{src['excerpt'][:100]}...")