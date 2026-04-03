# ADHD Psychoeducation Chatbot

A chatbot that helps users understand ADHD through trusted, 
citation-backed answers grounded in verified clinical sources.

**Live demo:** [your-app.vercel.app](https://your-app.vercel.app)  
**API docs:** [your-api.railway.app/docs](https://your-api.railway.app/docs)

---

## Why This Exists

General-purpose LLMs like ChatGPT can produce helpful answers, but in 
a healthcare context they have two critical limitations:

1. **Lack of source transparency** - you cannot verify where a specific 
   claim comes from.
2. **Uncontrolled knowledge** - responses may be outdated, inconsistent, 
   or not grounded in trusted clinical guidance.

For ADHD psychoeducation, this is not acceptable.

This system uses Retrieval-Augmented Generation (RAG) constrained to a 
curated knowledge base (CHADD, NIMH, CDC). Every response is generated 
from approved documents and includes citations to the exact source passage.

---

## Features

- **Hybrid retrieval** - semantic vector search + full-text keyword 
  search handles both natural language and exact terms like acronyms (RSD)
- **Reciprocal Rank Fusion** - combines search strategies reliably 
  without fragile score tuning
- **Cross-encoder reranking** - Cohere re-evaluates question and 
  context together for precise relevance
- **Grounded citations** - every response links to the specific source 
  passage with page number and excerpt
- **Confidence gate** - returns "I don't know" instead of hallucinating 
  answers outside the knowledge base
- **Healthcare guardrails** - crisis detection, scope enforcement, and 
  response sanitisation built in Python (not just system prompt)

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Database | PostgreSQL + pgvector |
| Embeddings | gemini-embedding-001(768 dims) |
| LLM | gemini-3.1-flash-lite-preview |
| Reranking | Cohere rerank-english-v3.0 |
| PDF extraction | PyMuPDF |
| Chunking | NLTK sentence-aware |
| Infrastructure | Docker (local) / Railway + Supabase (production) |
| Frontend | Vanilla HTML/CSS/JS on Vercel |

---

## Architecture
![Workflow](https://github.com/user-attachments/assets/fc465253-ea62-4f46-8f3e-ca724083b627)

---

## Running Locally

### Prerequisites
- Python 3.11
- Docker Desktop
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/adhd-chatbot.git
cd adhd-chatbot

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or: source venv/bin/activate  # Mac/Linux/WSL

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Variables
DATABASE_URL=postgresql://adhd_user:adhd_pass@localhost:5432/adhd_db
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key

Get your free API keys:
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Cohere: [cohere.com](https://cohere.com)

### Start the Database
```bash
docker compose up -d
python scripts/init_db.py
```

### Ingest Documents

Add your PDF files to `data/raw/pdf/`, then:
```bash
python scripts/chunk_documents.py
python scripts/ingest.py
```

### Start the API
```bash
uvicorn app.main:app --reload
```

API documentation available at: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health — DB + AI service status |
| GET | `/api/v1/stats` | Knowledge base statistics |
| GET | `/api/v1/demo` | Pre-loaded example questions |
| POST | `/api/v1/chat` | Main chatbot endpoint |
| POST | `/api/v1/feedback` | Submit answer feedback |

### Example Request
```bash
curl -X POST https://your-api.railway.app/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is rejection sensitive dysphoria?"}'
```

### Example Response
```json
{
  "answer": "Rejection sensitive dysphoria (RSD) is an intense emotional...",
  "session_id": "abc-123",
  "confidence": "high",
  "sources": [
    {
      "source_file": "chadd_emotional_regulation.pdf",
      "chunk_index": 4,
      "page_number": 2,
      "excerpt": "Rejection sensitive dysphoria refers to...",
      "relevance_score": 0.94,
      "confidence": "high"
    }
  ],
  "status": "success"
}
```

---

## Knowledge Base

Sources used (all publicly available, verified clinical material):
- CHADD (chadd.org) - Fact sheets on ADHD symptoms, adults, relationships
- NIMH (nimh.nih.gov) - ADHD overview and clinical summaries  
- CDC (cdc.gov/adhd) - Treatment and symptom guidance

---

## Disclaimer

This chatbot provides **educational information only**. It is not a 
substitute for professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare professional for personal 
medical decisions.

---

## Project Context

Built as an independent intern project demonstrating production RAG 
architecture on a real healthcare use case. Developed over 10 days 
starting from zero knowledge of FastAPI, pgvector, and RAG.
