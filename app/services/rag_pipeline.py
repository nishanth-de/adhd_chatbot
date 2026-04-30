import logging
import time
from typing import Generator
from app.services.retrieval import hybrid_search
from app.services.reranker import rerank_chunks, get_overall_confidence
from app.services.citations import build_citations
from app.services.llm import generate_answer, generate_answer_stream
from app.services.gaurdrails import check_input, sanitise_response

logger = logging.getLogger(__name__)

def run_rag_pipeline(question: str) -> dict:
    """
    Orchestrates the complete RAG pipeline with gaurdrails:
    0. Pre-LLM gaurdrail check(crisis, out_of_scope, conversational)
    1. Hybrid search (vector + FTS + RRF)
    2. Cohere reranking
    3. LLM Answer generation
    4. Post LLM answer sanitisation
    5. Citation building

    This is the single function your chat endpoint calls.
    Returns a structured dict matching ChatResponse schema.

    Arguments:
        question: user's natural language question

    Returns:
        dict with: answer, confidence, sources, retrieved_count
    """
    t0 = time.time()

    logger.info(f"RAG pipeline start | question='{question[:80]}'")

    # Stage 0: Gaurdrail check 
    # Runs before any API calls - Fast, deterministic and no cost
    gaurdrail_result = check_input(question)
    t1 = time.time()
    logger.info(f"Timming | Gaurdrail {t1 - t0:.2f} seconds")

    if not gaurdrail_result["safe"]:
        reason = gaurdrail_result["reason"]
        logger.info(f"Gaurdrail Blocked request | reason = {reason}")

        # Crisis gets a special confidence label
        confidence = "crisis" if reason == "crisis_detected" else "conversation" if reason == "conversational" else "out_of_scope"

        return{
        "answer": gaurdrail_result["blocked_response"],
        "confidence": confidence,
        "sources": [],
        "retrieved_count": 0
        }

    # Stage 1: Hybrid retrieval
    hybrid_chunks = hybrid_search(question, top_n=5)
    t2 = time.time()
    logger.info(f"Timing | hybrid_search: {t2-t1:.2f}s")

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
    t3 = time.time()
    logger.info(f"Timing | reranking: {t3-t2:.2f}s")

    # Stage 3: Confidence assessment
    confidence = get_overall_confidence(reranked_chunks)

    # Stage 4: Answer generation
    raw_answer = generate_answer(question, reranked_chunks)
    t4 = time.time()
    logger.info(f"Timing | generation: {t4-t3:.2f}s")

    # Stage 5: Post LLM-sanitisation
    # Catches unsafe content that bypases system prompt
    answer = sanitise_response(raw_answer)


    # Stage 6: Build citations
    citations = build_citations(reranked_chunks)

    logger.info(
        f"RAG pipeline complete | confidence={confidence} | "
        f"sources={len(citations)} | "
        f"sanitised={'yes' if answer != raw_answer else 'no'}"
    )

    logger.info(f"Timing | TOTAL: {t4-t0:.2f}s")

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": citations,
        "retrieved_count": len(hybrid_chunks)
    }



#                          run_rag_pipeline_stream

async def run_rag_pipeline_stream(question: str):
    """
    Streaming version of rag_pipeline.

    Run Retrieval synchronously (must complete before Generation starts), then Streams the
    Gemini Generation text tokens as they are produced.

    Yield:
        str: Either a JSON metadata prefix or text tokens

    Protocol:
        First Yield: JSON strings with citations and confidence
        Subsequent Yield: Answer(text token)
        This lets the frontend to show sources immediately while answer streams in.
    """
    import json

    # Stage 0: Gaurdrails(must complete anything else)
    gaurdrail_result = check_input(question)

    if not gaurdrail_result["safe"]:
        # Non streaming response for blocked messages
        # Yield as single chunk
        yield gaurdrail_result["blocked_response"]
        return
    
    # Stage 1: Retrieval(must complete before generation) - sync
    hybrid_chunks = hybrid_search(question, top_n=5)

    if not hybrid_chunks:
        yield(
            "I don't have reliable information about that in my knowledge base. "
            "For accurate ADHD information, please consult a healthcare professional "
            "or visit CHADD.org."
        )
        return
    
    # Stage 2: Reranking(must complete before generation) - sync
    top_similarity = hybrid_chunks[0].get("similarity", 0)
    if top_similarity >= 0.88:
        reranked_chunks = hybrid_chunks[:3]
        for chunk in reranked_chunks:
            chunk["relevance_score"] = top_similarity
            chunk["confidence"] = "high"
    else:
        reranked_chunks = rerank_chunks(question, hybrid_chunks, top_n=3)
    
    # Stage 3: Build citation (available before generation completes) - sync
    citations = build_citations(reranked_chunks)
    confidence = get_overall_confidence(reranked_chunks)

    # Yield metadata first as a JSON prefix
    # Frontend parses this to show sources immediately
    metadata = {
        "type": "metadata",
        "confidence": confidence,
        "sources": citations
    }
    yield f"METADATA:{json.dumps(metadata)}\n"

    # Stage 4 stream generation (True async, everything else is sync)
    async for token in generate_answer_stream(question, reranked_chunks):
        yield token



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