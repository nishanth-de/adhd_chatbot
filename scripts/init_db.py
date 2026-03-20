import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from sqlalchemy import text
from app.database import get_connection, test_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

def enable_vector_extension(conn):
    """
    Enables the pgvector extension in PostgreSQL.
    """
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    logger.info("pgvector extension enabled")


def create_documents_table(conn):
    """
    Creates the main documents table for storing text chunks and their embeddings.
    Columns:
    - id: auto-incrementing primary key
    - content: the actual text chunk from your ADHD documents
    - source: which file this chunk came from (for citations)
    - chunk_index: position of this chunk within its source document
    - embedding: the vector representation (768 dimensions for Gemini text-embedding-004)
    - content_tsv: preprocessed text for BM25 keyword search
    - created_at: when this chunk was ingested
    """
    # .execute() sends SQL to the database,it is a method of the SQLAlchemy connection object..
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          SERIAL PRIMARY KEY,

                -- CORE CONTENT
                content     TEXT NOT NULL,
                word_count  INTEGER NOT NULL DEFAULT 0,

                -- SOURCE TRACKING(powers grounded citations)
                source_file TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'unknown',
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,

                -- VECTOR EMBEDDING
                embedding   vector(768),

                -- BM25 KEYWORD SEARCH
                content_tsv  tsvector,

                -- TIMESTAMPS
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    )
    logger.info("documents table created with full metadata schema")



def create_indexes(conn):
    """
    Creates an HNSW index on the embedding column for fast similarity search.

    HNSW (Hierarchical Navigable Small World) 
    - It is an approximate nearest-neighbour algorithm. 
    - It trades a small amount of accuracy for large speed gains.
    - vector_cosine_ops tells pgvector to optimise for cosine distance (<=> operator).

    GIN index (content_tsv column):
    - Algorithm: Generalised Inverted Index
    - Purpose: fast keyword lookup for full-text / BM25 search
    - GIN is the standard index type for tsvector columns
    """
    conn.execute(
        text(
        """
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents
        USING hnsw (embedding vector_cosine_ops);
        """
        )
    )
    logger.info("HNSW vector index created (or already exists)")

    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS documents_tsv_idx
        ON documents
        USING gin (content_tsv);
    """))
    logger.info("GIN full-text search index created")



def verify_setup(conn):
    """
    Verifies the database is set up correctly by checking
    the extension and table exist.
    """
    # Check extension
    result = conn.execute(
        text(
        """
        SELECT extname FROM pg_extension WHERE extname = 'vector';
        """
        )
    )
    extension_name = result.fetchone()
    logger.info(f"pgvector extension: {'FOUND' if extension_name else 'NOT FOUND'}")

    # Check table
    result = conn.execute(
        text(
            """
            SELECT 
                table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'documents';
            """
        )
    )
    table = result.fetchone()
    logger.info(f"documents table: {'FOUND' if table else 'NOT FOUND'}")

    # Check both indexes exist
    result = conn.execute(text("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'documents'
        ORDER BY indexname;
    """))
    indexes = [row[0] for row in result.fetchall()]
    logger.info(f"Indexes found: {indexes}")

    # Check row count
    result = conn.execute(text("SELECT COUNT(*) FROM documents;"))
    count = result.fetchone()[0] 
        # .fetchone() -> gets first row. 
        # [0] -> gets first column.
    logger.info(f"documents table row count: {count}")


def init_db():
    """
    Main initialization function. Runs all setup steps in order.
    Safe to run multiple times — all operations use IF NOT EXISTS.
    """
    logger.info("Starting database initialization...")

    # First verify we can connect at all
    if not test_connection():
        logger.error("Cannot connect to database. Aborting initialization.")
        logger.error("Check: is Docker running? Try: docker ps")
        sys.exit(1)

    with get_connection() as conn:
        enable_vector_extension(conn)
        create_documents_table(conn)
        create_indexes(conn)
        conn.commit()  # commit all changes together
        verify_setup(conn)

    logger.info("Database initialization complete!")


if __name__ == "__main__":
    init_db()