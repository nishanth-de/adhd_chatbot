import logging

logger = logging.getLogger(__name__)

EXCERPT_LENGTH = 200  # characters to show in citation excerpt

def build_citations(reranked_chunks: list[dict]) -> list[dict]:
    """
    Converts reranked chunks into structured source citations
    for the API response.

    Citations will give users the ability to verify answers against
    the original source — This is our Edge and critical for a healthcare application.

    Each citation includes:
    - source_file: which PDF the chunk came from
    - chunk_index: position within that document
    - page_number: page in the original PDF (for manual lookup)
    - excerpt: first N characters of the chunk (identifies the passage)
    - relevance_score: how relevant Cohere scored this chunk
    - confidence: human-readable confidence level

    Arguments:
        reranked_chunks: output of rerank_chunks()

    Returns:
        list of citation dicts ready for API serialisation
    """
    citations = []

    for chunk in reranked_chunks:
        content = chunk.get("content", "")

        # Creating a clean excerpt — first EXCERPT_LENGTH chars
        # Strip leading whitespace for extra cleaning!!
        excerpt = content.strip()[:EXCERPT_LENGTH]

        # If we cut mid-sentence, add ellipsis
        if len(content.strip()) > EXCERPT_LENGTH:
            excerpt += "..."

        citation = {
            "source_file": chunk.get("source_file", "unknown"),
            "chunk_index": chunk.get("chunk_index", 0),
            "page_number": chunk.get("page_number", 0),
            "excerpt": excerpt,
            "relevance_score": chunk.get("relevance_score", 0.0),
            "confidence": chunk.get("confidence", "low")
        }
        citations.append(citation)        

    logger.info(f"Built {len(citations)} citations")
    return citations

def format_context_for_llm(reranked_chunks: list[dict]) -> str:
    """
    Formats reranked chunks into a context string for the LLM prompt.

    Each chunk is labelled with its source so the LLM can reference
    it in its answer if needed.

    Arguments:
        reranked_chunks: output of rerank_chunks()

    Returns:
        formatted context string to inject into LLM prompt
    """
    if not reranked_chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(reranked_chunks, start=1):
        source = chunk.get("source_file", "unknown")
        content = chunk.get("content", "")
        context_parts.append(
            f"[Source {i}: {source}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    # Test with fake reranked chunks
    fake_chunks = [
        {
            "content": "Executive dysfunction is one of the core challenges "
                        "of ADHD. It affects planning, task initiation, and "
                        "working memory. People with ADHD often know what they "
                        "need to do but cannot get started.",
            "source_file": "chadd_executive_function.pdf",
            "chunk_index": 3,
            "page_number": 2,
            "relevance_score": 0.92,
            "confidence": "high"
        },
        {
            "content": "ADHD affects approximately 5-7% of children and "
                        "2-5% of adults worldwide. It is characterised by "
                        "persistent inattention, hyperactivity, and impulsivity.",
            "source_file": "nimh_adhd_overview.pdf",
            "chunk_index": 1,
            "page_number": 1,
            "relevance_score": 0.74,
            "confidence": "high"
        }
    ]

    print("=== Citation Builder Test ===\n")

    citations = build_citations(fake_chunks)

    print("Citations:")
    for i, c in enumerate(citations, 1):
        print(
            f"\n[{i}] {c['source_file']} - chunk {c['chunk_index']}, "
            f"page {c['page_number']}"
        )
        print(f"Relevance: {c['relevance_score']} ({c['confidence']})")
        print(f"Excerpt: {c['excerpt']}")

    print("\n\nContext for LLM:")
    print(format_context_for_llm(fake_chunks))