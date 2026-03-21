import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from sqlalchemy import text
from app.database import get_connection, test_connection
from app.services.embeddings import get_embedding


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)



def chunk_already_exists(source_file: str, chunk_index: int) -> bool:
    """
    Checks if a chunk has already been ingested.
    This our idempotency check — makes the script safe to re-run.

    Args:
        source_file: the PDF filename
        chunk_index: position of chunk within that file

    Returns:
        True if already in database, False if not
    """
    with get_connection() as conn:
        result = conn.execute(
            text("""
                SELECT id 
                FROM documents
                WHERE source_file = :source_file
                    AND chunk_index = :chunk_index
            """),
            {"source_file": source_file, "chunk_index": chunk_index}
        )
        return result.fetchone() is not None



def ingest_single_chunk(chunk: dict) -> bool:
    """
    Embeds and stores a single chunk in the DB.
    Each chunk gets its own transaction (committed immediately).
    Why?
    If a failure happens it affects only a chunk, not others.

    Args:
        chunk: dict with keys: content, source_file, source_type,
        page_number, chunk_index, word_count.

    Returns:
        True if successfully ingested, else False
    """
    source_file = chunk["source_file"]
    chunk_index = chunk["chunk_index"]

    # Step 1: Idempotency check
    if chunk_already_exists(source_file, chunk_index):
        logger.info(f"Skipping (already exists): {source_file} chunk {chunk_index}")
        return True
    
    # Step 2: Generate embedding via Gemini
    # get_embedding() already has retry logic for rate limits!! 
    embedding = get_embedding(chunk["content"])

    # Step 3: Insert into database
    # Note: to_tsvector('english', :content) is PostgreSQL generating
    # the BM25 index — we don't do this in Python
    with get_connection() as conn:
        conn.execute(
            text("""
                INSERT INTO documents (
                    content,
                    word_count,
                    source_file,
                    source_type,
                    page_number,
                    chunk_index,
                    embedding,
                    content_tsv
                ) VALUES (
                    :content,
                    :word_count,
                    :source_file,
                    :source_type,
                    :page_number,
                    :chunk_index,
                    :embedding,
                    to_tsvector('english', :content) -- Refer notes!!
                )
            """),
            {
                "content":     chunk["content"],
                "source_file": source_file,
                "source_type": chunk.get("source_type", "pdf"),
                "page_number": chunk.get("page_number", 0),
                "chunk_index": chunk_index,
                "word_count":  chunk["word_count"],
                "embedding":   str(embedding)  # pgvector accepts "[0.1, 0.2, ...]"
            }
        )
        conn.commit()  # commit immediately for each chunk, so this chunk is now safe!!

    logger.info(
        f"Ingested: {source_file} chunk {chunk_index} "
        f"({chunk['word_count']} words)"
    )
    return True



def run_ingestion(chunks_path: str = r"data/processed/all_chunks.json") -> dict:
    """
    Main ingestion loop. Processes all chunks with:
    - Idempotency (skips already-ingested chunks)
    - Per-chunk error handling (one chunk failure doesn't stop others)
    - Rate limit courtesy sleep between API calls
    - Progress logging every 10 chunks

    Args:
        chunks_path: path to the JSON file produced by chunk_documents.py

    Returns:
        dict with ingestion statistics
    """

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    total = len(chunks)
    logger.info(f"Starting ingestion of {total} chunks from {chunks_path}")

    statistics = {
        "total": total,
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "failed_chunks": []  # tracking which one was failed for debugging
    }

    for i, chunk in enumerate(chunks):
        source = chunk.get("source_file", "unknown")
        index = chunk.get("chunk_index", i)

        try:
            # Idempotency check happens inside ingest_single_chunk
            if chunk_already_exists(source, index):
                statistics["skipped"] += 1
                logger.info(f"[{i+1}/{total}] Skipped (exists): {source} chunk {index}")
                continue
            # Embed and insert
            success = ingest_single_chunk(chunk)

            if success:
                statistics["ingested"] += 1
            else:
                statistics["failed"] += 1
                statistics["failed_chunks"].append(f"{source}:{index}")

        except Exception as e:
            # This chunk failed — log it, record it, continue to next
            # The exception does NOT propagate — other chunks are unaffected
            statistics["failed"] += 1
            statistics["failed_chunks"].append(f"{source}:{index}")
            logger.error(
                f"[{i+1}/{total}] FAILED: {source} chunk {index} — {e}"
            )

        finally:
            # Rate limit courtesy — runs whether chunk succeeded or failed
            # Keeps us well under Gemini free tier limits
            time.sleep(0.1)     

        # Progress update every 10 chunks
        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{total} | "
                f"Ingested: {statistics['ingested']} | "
                f"Skipped: {statistics['skipped']} | "
                f"Failed: {statistics['failed']}"
            )

    return statistics

def verify_ingestion():
    """
    Post-ingestion verification. Checks that:
    1. Row count matches expected
    2. All rows have embeddings
    3. All rows have tsvector populated
    4. Source distribution looks correct
    """
    print("\n=== INGESTION VERIFICATION ===")

    with get_connection() as conn:
        # Total rows
        result = conn.execute(text("SELECT COUNT(*) FROM documents;"))
        total = result.fetchone()[0]
        print(f"Total rows in database: {total}")


        # Rows with embeddings
        result = conn.execute(
            text("""
                SELECT 
                    COUNT(*) 
                FROM documents 
                WHERE embedding IS NOT NULL;
            """)
        )
        with_embedding = result.fetchone()[0]
        print(f"Rows with embeddings: {with_embedding}")

        # Rows with tsvector populated
        result = conn.execute(
            text("""
                SELECT 
                    COUNT(*) 
                FROM documents
                WHERE content_tsv IS NOT NULL;
            """)
        )
        with_tsv = result.fetchone()[0]
        print(f"Rows with tsvector (BM25 ready): {with_tsv}")


        # Source distribution
        result = conn.execute(
            text("""
                SELECT 
                    source_file, 
                    COUNT(*) as chunks
                FROM documents
                GROUP BY source_file
                ORDER BY chunks DESC;
            """)
        )
        print(f"\nChunks per source:")
        for row in result.fetchall():
            print(f"  {row[0]}: {row[1]} chunks")


        # Quick sanity — fetch one embedding and check dimensions
        result = conn.execute(
            text("""
                SELECT 
                    content,
                    embedding
                FROM documents
                WHERE embedding IS NOT NULL
                LIMIT 1;
            """)
        )
        row = result.fetchone()
        if row:
            # pgvector returns embedding as a string like "[0.1, 0.2, ...]"
            # Parse it to verify dimensions
            emb_str = str(row[1])
            dims = len(emb_str.strip('[]').split(','))
            print(f"\nEmbedding dimension check: {dims} (expected 768)")



if __name__ == "__main__":
    print("=== ADHD Chatbot — Document Ingestion Pipeline ===\n")

    # Pre-flight checks
    if not test_connection():
        print("ERROR: Cannot connect to database. Is Docker running?")
        sys.exit(1)

    if not os.path.exists("data/processed/all_chunks.json"):
        print("ERROR: data/processed/all_chunks.json not found.")
        print("Run: python scripts/chunk_documents.py first")
        sys.exit(1)
    
    # Run ingestion
    stats = run_ingestion()

    # Print final stats
    print(f"\n=== INGESTION COMPLETE ===")
    print(f"Total chunks:    {stats['total']}")
    print(f"Newly ingested:  {stats['ingested']}")
    print(f"Skipped (exist): {stats['skipped']}")
    print(f"Failed:          {stats['failed']}")

    if stats['failed_chunks']:
        print(f"\nFailed chunks (investigate these):")
        for failed_chunks in stats['failed_chunks']:
            print(f"  {failed_chunks}")
    
    # Verifying what's in the database
    verify_ingestion()

    if stats['failed'] == 0:
        print("\n Ingestion successful — knowledge base is live")
    else:
        print(f"\n Ingestion completed with {stats['failed']} failures")
        print("Re-run this script to retry failed chunks (idempotent)")