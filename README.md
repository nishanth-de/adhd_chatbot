# ADHD Psychoeducation Chatbot
Chatbot that helps users understand ADHD through trusted, citation-backed answers.

# How it works
This is an ADHD psychoeducation chatbot built on a production RAG architecture.
Every answer comes from a curated knowledge base of verified sources — CHADD, NIMH, and CDC documents.
The system uses hybrid retrieval combining vector semantic search and full-text keyword search, fused with Reciprocal Rank Fusion, then reranked by a Cohere cross-encoder. 
Every response includes grounded citations so users can verify answers against the original source, which matters in a healthcare context.

## Why

**Why not just use ChatGPT or Gemini directly?**

General-purpose LLMs can produce helpful answers, but in a healthcare context they have two critical limitations:

1. **Lack of source transparency** — you cannot verify where a specific claim comes from.
2. **Uncontrolled knowledge** — responses may be outdated, inconsistent, or not grounded in trusted clinical guidance.

For ADHD psychoeducation, this is not acceptable.

This system uses a Retrieval-Augmented Generation (RAG) approach to constrain the model to a curated knowledge base (CHADD, NIMH, CDC). 
Every response is generated from approved documents and includes citations to the original source.

This ensures responses are:
- Grounded in trusted material
- Traceable and verifiable
- Restricted to safe, non-speculative guidance

This is the difference between a general chatbot and a system designed for reliable psychoeducation.

## Features

- **Hybrid retrieval (semantic + keyword search)** — handles both natural language queries and exact terms like acronyms (e.g., RSD)
- **Reciprocal Rank Fusion (RRF)** — combines multiple search strategies reliably without fragile score tuning
- **Cross-encoder reranking** — improves relevance by evaluating the question and context together
- **Grounded citations** — every response links to a specific, verifiable source passage
- **Out-of-scope detection** — avoids answering when the query is outside the knowledge base
- **Designed for reliability over raw generation** — prioritizing precision, traceability, and safety in a healthcare context.

## Technical Architecture

- **API layer** — FastAPI with auto-generated OpenAPI docs (`/docs`)
- **Database** — PostgreSQL with pgvector for vector storage and retrieval
- **Retrieval pipeline**
  - Hybrid search: semantic (embeddings) + keyword (full-text search)
  - Fusion using Reciprocal Rank Fusion (RRF)
- **Reranking** — Cohere `rerank-english-v3.0` cross-encoder
- **LLM generation** — Gemini 3.1 Flash with a constrained healthcare system prompt
- **Citations** — responses include source file, page number, excerpt, and relevance metadata
