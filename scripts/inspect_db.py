"""
Building an utility script that let's us to inspect our DB state at any time.
"""

import sys
import os
# Allows importing modules from the parent project folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.database import get_connection, test_connection

def show_table_stats():
    """Shows row counts and source distribution in documents table."""
    print("\n=== TABLE STATS ===")
    with get_connection() as conn:
        # Total rows
        result = conn.execute(text("SELECT COUNT(*) FROM documents;"))
        total = result.fetchone()[0]
        print(f"Total chunks: {total}")

        if total > 0:
            # Breakdown by source
            result = conn.execute(
                            text(
                                """
                                SELECT source, COUNT(*) as chunk_count
                                FROM documents
                                GROUP BY source
                                ORDER BY chunk_count DESC;
                                """
                            )
                        )
            print("\nChunks per source:")
            for source, chunk_count in result.fetchall():
                print(f"{source[0]}: {chunk_count} chunks")

def show_sample_chunks(n: int = 3):
    """Shows the first N chunks stored in the database."""
    print(f"\n=== SAMPLE CHUNKS (first {n}) ===")
    with get_connection() as conn:
        result = conn.execute(text
                        (f"""
                        SELECT  
                            id, 
                            source, 
                            chunk_index,
                            LEFT(content, 200) as content_preview,
                            CASE WHEN embedding IS NULL THEN 'NO' ELSE 'YES' END as has_embedding
                        FROM documents
                        ORDER BY id
                        LIMIT {n};
                        """)
        )
        rows = result.fetchall()

        if not rows:
            print("No chunks found. Run ingestion first.")
            return

        for row in rows:
            print(f"\nID: {row[0]} | Source: {row[1]} | Chunk: {row[2]}")
            print(f"Has embedding: {row[4]}")
            print(f"Content preview: {row[3]}...")

def show_embedding_stats():
    """Shows embedding coverage - how many chunks havs embeddings"""
    print("\n===EMBEDDING STATS ===")
    with get_connection() as connection:
        result = connection.execute(
            text(
                """
                SELECT
                    COUNT(*) AS Total,
                    COUNT(embedding) AS with_embedding,
                    COUNT(*) - COUNT(embedding) as without_embedding
                FROM documents;    
                """
            )
        )
    row = result.fetchone()
    print(f"Total chunks: {row[0]}")
    print(f"With embeddings: {row[1]}")
    print(f"Without embeddings: {row[2]}")

def custome_query(query: str):
    """
    Runs a custom SQL Query and prints results.
    """
    print(f"Cutomer query: {query}")
    with get_connection() as connection:
        result = connection.execute(text(query))
        rows = result.fetchall()
        if rows:
            for row in rows:
                print(row)
        else:
            print("No results.")

if __name__ == "__main__":
    if not test_connection():
        print("Cannot connect to the Database! is docker running?")
        sys.exit(1)

    show_table_stats()
    show_sample_chunks(3)
    show_embedding_stats()    