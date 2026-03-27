# Building Vector and Text search

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from sqlalchemy import text
from database import get_connection
from services.embeddings import get_query_embedding

logger = logging.getLogger(__name__)

# Minimum similarity for a result to be considered relevant
# Below this threshold -> knowledge base has no good answer.
SIMILARITY_THRESHOLD = 0.60

# How many results to fetch from each search method
# NOTE: Always Fetch more than we need — reranking will trim later
RETRIEVAL_TOP_K = 10

def vector_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """
    Semantic vector search using pgvector cosine similarity.

    Converts query to embedding, finds most similar chunks.
    Returns empty list if best match is below similarity threshold
    (signals to LLM: no reliable information found).

    Arguments:
        query: user's natural language question
        top_k: number of results to return

    Returns:
        list of dicts with content, source_file, chunk_index,
        similarity, and search_type="vector"
    """
    logger.info(f"Vector search | query='{query[:60]}...' | top_k={top_k}")

    # Converting user question to embedding using RETRIEVAL_QUERY task type
    query_embedding = get_query_embedding(query)

    with get_connection() as conn:
        result = conn.execute(
            text("""
                SELECT
                    id,
                    content,
                    source_file,
                    chunk_index,
                    page_number,
                    word_count,
                    1 - (embedding <=> (:query_vector)::vector) AS similarity
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> (:query_vector)::vector
                LIMIT :top_k
            """),
            {
                "query_vector": str(query_embedding), 
                "top_k": top_k
            }
        )
        rows = result.fetchall()

    if not rows:
        logger.warning("Vector search returned no results")
        return []
        
    results = [
        {
            "id": row[0],
            "content": row[1],
            "source_file": row[2],
            "chunk_index": row[3],
            "page_number": row[4],
            "word_count": row[5],
            "similarity": round(float(row[6]), 4),
            "search_type": "vector"
        }
        for row in rows
    ]

    # Log top result for monitoring
    top = results[0]
    logger.info(
        f"Vector search top result | "
        f"similarity={top['similarity']} | "
        f"source={top['source_file']} chunk {top['chunk_index']}"
    )

        # Confidence threshold check
    if results[0]['similarity'] < SIMILARITY_THRESHOLD:
        logger.warning(
            f"Top similarity {results[0]['similarity']} below threshold "
            f"{SIMILARITY_THRESHOLD} — query may be out of scope"
        )
        # Still return results — caller decides whether to proceed
        # Threshold enforcement happens in the hybrid search layer

    return results

def bm25_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """
    BM25 keyword search using PostgreSQL full-text search.

    Keyword search using PostgreSQL full-text search (tsvector/tsquery).
    Uses ts_rank for relevance scoring — functionally similar to BM25
    for short documents, though not the BM25 algorithm specifically.
    True BM25 would require the pg_bm25 extension (ParadeDB).

    Uses websearch_to_tsquery for natural language input —
    handles multi-word queries, punctuation, and user phrasing
    without requiring operators like & or |.

    Returns empty list if no chunks contain the query terms.
    This is expected and correct — BM25 is precise, not approximate.

    Args:
        query: user's natural language question
        top_k: number of results to return

    Returns:
        list of dicts with content, source_file, chunk_index,
        bm25_score, and search_type="bm25"
    """
    logger.info(f"BM25 search | query='{query[:60]}' | top_k={top_k}")

    with get_connection() as conn:
        result = conn.execute(
            text("""
                SELECT
                    id,
                    content,
                    source_file,
                    chunk_index,
                    page_number,
                    word_count,
                    ts_rank(content_tsv,
                        websearch_to_tsquery('english', :query)
                    ) AS bm25_score
                FROM documents
                WHERE content_tsv @@ websearch_to_tsquery('english', :query)
                ORDER BY bm25_score DESC
                LIMIT :top_k
            """),
            {
                "query": query,
                "top_k": top_k
            }
        )
        rows = result.fetchall()

    if not rows:
        logger.info(f"BM25 search: no keyword matches for '{query[:60]}'")
        return []

    results = [
        {
            "id": row[0],
            "content": row[1],
            "source_file": row[2],
            "chunk_index": row[3],
            "page_number": row[4],
            "word_count": row[5],
            "bm25_score": round(float(row[6]), 4),
            "search_type": "bm25"
        }
        for row in rows
    ]

    logger.info(
        f"BM25 search: {len(results)} results | "
        f"top score={results[0]['bm25_score']} | "
        f"source={results[0]['source_file']}"
    )
    return results

if __name__ == "__main__":
    print("=== Vector Search Test ===\n")

    test_queries = [
        "What is ADHD?",
        "Why do people with ADHD struggle to start tasks?",
        "What is rejection sensitive dysphoria?",
        "How does working memory affect ADHD?",
        "What is the capital of France?",  # deliberately off-topic query
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")

        # Vector results
        vector_results = vector_search(query, top_k=3)
        print(f"\nVector Search ({len(vector_results)} results):")
        for i, r in enumerate(vector_results[:3]):
            print(f"\n[{i+1}] sim={r['similarity']} | "
                f"{r['source_file']} chunk {r['chunk_index']}")

        # BM25 results
        bm25_results = bm25_search(query, top_k=3)
        print(f"\nBM25 Search ({len(bm25_results)} results):")
        if not bm25_results:
            print("No keyword matches found")
        for i, r in enumerate(bm25_results[:3]):
            print(f"\n[{i+1}] score={r['bm25_score']} | "
                f"{r['source_file']} chunk {r['chunk_index']}")