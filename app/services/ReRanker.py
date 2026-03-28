import os
import logging
from dotenv import load_dotenv
import cohere

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found. Check your .env file.")

# Initialise client once at module level
co = cohere.ClientV2(api_key=api_key)

RERANK_MODEL = "rerank-english-v3.0"

# Relevance thresholds for confidence classification
HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.40


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 3
) -> list[dict]:
    """
    Reranks retrieved chunks using Cohere's cross-encoder model.

    Takes the output of hybrid_search() and re-scores each chunk
    by jointly encoding the query and chunk content together.

    Why??
    This catches relevance that RRF missed because RRF only uses
    rank positions, not actual content.

    Arguments:
        query: the user's original question
        chunks: list of chunk dicts from hybrid_search()
        top_n: how many reranked chunks to return

    Returns:
        reranked list of top_n chunks with relevance_score added
        Returns input chunks unchanged if reranking fails (graceful fallback)
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        # No point reranking a single chunk
        chunks[0]["relevance_score"] = 0.5
        chunks[0]["confidence"] = "medium"
        return chunks
    
    logger.info(
        f"Reranking {len(chunks)} chunks | "
        f"query='{query[:60]}' | top_n={top_n}"
    )

    # Extract just the text content for Cohere
    # Cohere's rerank takes plain strings, not dicts
    documents = [chunk["content"] for chunk in chunks]

    try:
        response = co.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_n
        )

        # Cohere returns results with:
        # -> .index: which document in our list (0-based)
        # -> .relevance_score: float 0-1, higher = more relevant
        reranked = []
        for result in response.results:
            # Use the index to get the original chunk dict back
            original_chunk = chunks[result.index].copy()
            original_chunk["relevance_score"] = round(result.relevance_score, 4)
            original_chunk["confidence"] = classify_confidence(result.relevance_score)
            reranked.append(original_chunk)

        logger.info(
            f"Reranking complete | "
            f"top score={reranked[0]['relevance_score']} | "
            f"confidence={reranked[0]['confidence']}"
        )
        return reranked

    except Exception as e:
        # Reranking failed fall back to original RRF ordering
        # This is a graceful degradation: system still works, just less precise
        logger.error(f"Reranking failed, using RRF order: {e}")
        for chunk in chunks[:top_n]:
            chunk["relevance_score"] = 0.5  # neutral score
            chunk["confidence"] = "medium"
        return chunks[:top_n]   

def classify_confidence(relevance_score: float) -> str:
    """
    Converts a numeric relevance score into a human-readable confidence level.

    Thresholds are calibrated for healthcare context — we prefer
    saying "I don't know" over giving a low-confidence answer.

    Arguments:
        relevance_score: float 0-1 from Cohere reranker

    Returns:
        "high", "medium", or "low"
    """
    if relevance_score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif relevance_score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    else:
        return "low"
    
def get_overall_confidence(reranked_chunks: list[dict]) -> str:
    """
    Derives overall response confidence from the top reranked chunk.

    If the best chunk is low confidence, the whole response is low confidence
    regardless of what other chunks scored.

    Arguments:
        reranked_chunks: output of rerank_chunks()

    Returns:
        "high", "medium", "low", or "out_of_scope"
    """
    if not reranked_chunks:
        return "out_of_scope"
    return reranked_chunks[0]["confidence"]

if __name__ == "__main__":

    from app.services.retrieval import hybrid_search

    print("=== Reranker Service Test ===\n")

    test_queries = [
        "What is rejection sensitive dysphoria?",
        "How does ADHD affect executive function?",
        "What is the capital of Tamil Nadu?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Step 1: Hybrid search
        chunks = hybrid_search(query, top_n=5)

        if not chunks:
            print("No chunks retrieved (confidence gate triggered)")
            continue

        print(f"Hybrid search: {len(chunks)} chunks")

        # Step 2: Rerank
        reranked = rerank_chunks(query, chunks, top_n=3)

        print(f"After reranking: {len(reranked)} chunks")
        for i, chunk in enumerate(reranked):
            print(
                f"[{i+1}] relevance={chunk['relevance_score']} | "
                f"confidence={chunk['confidence']} | "
                f"{chunk['source_file']} chunk {chunk['chunk_index']}"
            )
            print(f"{chunk['content'][:100]}...")