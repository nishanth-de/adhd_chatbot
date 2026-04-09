# Building Vector and Text search

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from sqlalchemy import text
from app.database import get_connection
from app.services.embeddings import get_query_embedding

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

    Arguments:
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



def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_n: int = 5
) -> list[dict]:
    """
    Combines vector and BM25 results using Reciprocal Rank Fusion(RRF).

    RRF_score = sum of: 1 / (rank_in_that_list + 60)

    Where rank is 1-indexed position in each result list.
    k=60 is the standard constant — to prevent top-rank dominance.

    Chunks appearing in both lists get scores from both,
    making them rank higher than chunks in only one list.

    Args:
        vector_results: ranked list from vector_search()
        bm25_results: ranked list from bm25_search()
        k: RRF constant (default 60)
        top_n: how many results to return after fusion

    Returns:
        fused ranked list of top_n chunks with rrf_score
    """
    # Using chunk id as the unique identifier across both lists
    rrf_scores: dict[int, float] = {} # Accumulate the running RRF score for each chunk ID
    chunk_data: dict[int, dict] = {} # stores the full chunk dictionary

    # Scoring vector results
    # enumerate gives (0-indexed position, item) — add 1 for 1-indexed rank
    for rank, chunk in enumerate(vector_results, start=1):
        chunk_id = chunk["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + k)
        chunk_data[chunk_id] = chunk  # store full chunk data

    # Score BM25 results — add to existing score if chunk already seen
    for rank, chunk in enumerate(bm25_results, start=1):
        chunk_id = chunk["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + k)
        # Only store if not already stored (vector result takes precedence
        # for data, since it has similarity score)
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = chunk

    # Sorting by RRF score in descending order
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Building the final results list
    fused = []
    for chunk_id in sorted_ids[:top_n]:
        chunk = chunk_data[chunk_id].copy()
        chunk["rrf_score"] = round(rrf_scores[chunk_id], 6)
        fused.append(chunk)
    
    logger.info(
        f"RRF fusion | vector={len(vector_results)} + "
        f"bm25={len(bm25_results)} → top {len(fused)}"
    )
    return fused    


# Unified hybrid search function, this function is called by our route services!! 
def hybrid_search(query: str, top_n: int = 5) -> list[dict]:
    """
    Main retrieval function — combines vector and BM25 search with RRF.

    This is the only function the rest of the application should call.
    Internal implementation (vector search, BM25, RRF) is encapsulated here.

    Workflow:
    1. Run vector search (semantic similarity)
    2. Run BM25 search (keyword matching)
    3. Fuse results with RRF
    4. Apply confidence gate — return empty if query is out of scope
    5. Return top_n chunks ready for the LLM

    Arguments:
        query: user's natural language question
        top_n: number of chunks to return to the LLM (default 5)

    Returns:
        list of top_n chunks, or empty list if query is out of scope
    """
    logger.info(f"Hybrid search | query='{query[:80]}'")

    # Run both searches in sequence!!
    # Note: in a production system we need to run these concurrently
    vector_results = vector_search(query, top_k=10)
    bm25_results = bm25_search(query, top_k=10)

    # If both searches return nothing — definitely out of scope
    if not vector_results and not bm25_results:
        logger.warning(f"Both searches returned nothing for: '{query[:60]}'")
        return []
    
    # Fusing results with RRF
    fused_results = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_n=top_n
    )

    # Confidence gate — check for vector similarity top results
    # Only vector results have similarity scores — and we use that as proxy
    # for overall confidence
    top_vector_results = [r for r in fused_results if 'similarity' in r]
    if top_vector_results:
        top_similarity = top_vector_results[0]['similarity']
        if top_similarity < SIMILARITY_THRESHOLD:
            logger.warning(
                f"Confidence gate triggered | "
                f"top_similarity={top_similarity} < {SIMILARITY_THRESHOLD} | "
                f"query='{query[:60]}'"
            )
            return []  # LLM service will return "I don't know" response
    
    logger.info(
        f"Hybrid search complete | "
        f"returning {len(fused_results)} chunks"
    )
    return fused_results

if __name__ == "__main__":
    print("=== Vector Search Test ===\n")

    test_queries = [
        "What is ADHD?",
        "Why do people with ADHD struggle to start tasks?",
        "What is rejection sensitive dysphoria?",
        "How does working memory affect ADHD?",
        "What is the capital of France?",  # deliberate off-topic query
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")

        results = hybrid_search(query, top_n=4)

        if not results:
            print("Confidence gate triggered: query out of scope")
            print("LLM will respond: 'I don't have reliable info on that'")
            continue

        print(f" {len(results)} chunks retrieved\n")
        for i, r in enumerate(results):
            print(f"[{i+1}] RRF={r.get('rrf_score', 'N/A')} | "
                f"sim={r.get('similarity', 'N/A')} | "
                f"{r['source_file']} chunk {r['chunk_index']}")
            print(f" {r['content'][:120]}...")