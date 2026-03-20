# PDF Text Extractor
# Writing a Function that reads a PDF and extracts clean text.

import os
import sys
import logging
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
# used to split a given text or paragraph into a list of individual sentences. 
# It leverages a pre-trained model (PunktSentenceTokenizer).
# to intelligently identify sentence boundaries based on punctuation, capitalization, 
# and other linguistic patterns, handling complex cases like abbreviations.
"""
import nltk
from nltk.tokenize import sent_tokenize 
import fitz # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format= "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# Downloading required data if not already present
# Safe to call repeatedly - only downloads if missing

# Punkt handles tokenization (splitting text)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)



# Footer patterns to strip
BOILERPLATE_PATTERNS = [
    # NICE guidelines footer — matches any guideline code and year
    # Example: "(NG87) © NICE 2025. All rights reserved. Subject to Notice of rights (URL)"
    r'\([A-Z]+\d+\)\s*©\s*NICE\s*\d{4}\.?\s*All rights reserved\..*?(?=\n|$)',

    # Generic copyright lines
    r'©\s*\d{4}.*?All rights reserved\.?',

    # URLs on their own line (often footers)
    r'https?://\S+',

    # "Page X of Y" patterns
    r'Page\s+\d+\s+of\s+\d+',
]

def remove_boilerplate(text: str) -> str:
    """
    Removes footer/header patterns from extracted text.
    These are legal notices, page numbers, URLs, and repeated headers
    which is noise to our embeddings and adds no information.
    """
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Clean up any blank lines created by removals
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and cleans text from a pdf file.

    Cleaning steps:
        -> Extracts text page by page
        -> Remove exessice white space
        -> Fix hyphenated line breaks (re-join split words)
        -> Remove page numbers and common headers/footers patterns
    
    Args:
        pdf_path: Full path of the pdf file
    
    Returns: 
        Cleaned text string
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
    
        # Fix hyphenated line breaks
        # Eg: execu-\ntive -> executive
        text = re.sub(r'-\n', "", text)

        # Replacing single new line with space (paragraph flow)
        # But preserve double new lines (paragraph break)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Removing excessive whitespaces
        text = re.sub(r' +', " ", text)

        # Removing standalone pagenumbers (eg: (e.g., "- 3 -" or just "3" on a line))
        text = re.sub(r"\n\s*-?\s*\d+\s*-?\s*\n", "\n", text)

        pages_text.append(text.strip())

    doc.close()

    # Joining all pages
    full_text = "\n\n".join(pages_text)

    # Final cleanup — collapse more than 2 consecutive newlines
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Remove boilerplate AFTER joining all pages
    # Why after? Because some footers span a line break across the join point
    full_text = remove_boilerplate(full_text)

    return full_text.strip()



# Building sentence aware chunker
# Writing the chunking function that splits text at sentence boundaries.
# But Why? Chunks that end mid-sentence will produce worst embeddings
# Mental Model: Better chunks = Better retrieval = Better Answer
def chunk_text_by_sentece(
        text: str,
        source: str,
        target_chunk_words: int = 300,
        overlap_sentences: int = 2
    ) -> list[dict]:
    """
    Splits texts into chunks at sentence boundaries
    
    Algorithm:
        -> Using nltk to detect sentence boundaries
        -> Accumulate sentences until target words count is reached
        -> Start new chunk, carrying last "overlap_sentence" into it
        -> Each chunk is a complete set of sentence, no mid-sentence
    
    Args:
        -> text: Cleaned text to chunk.
        -> source: Filename of the source pdf(for citations).
        -> target_chunk_words: approximate target words per chunk.
        -> overlap_sentences: sentence to carry over between chunks

    Returns:
        list of dict with content, source, chunk_index, word_count
    """
    # sent_tokenize(text) Splits a document into a list of sentences
    # using the punkt_tab model to identify where sentences
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]

    if not sentences:
        logger.warning("No sentence detected in {source}")
        return[]

    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += sentence_words

        # When we hit target size we save the chunks
        if current_word_count >= target_chunk_words:
            chunk_text = " ".join(current_sentences)

            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
                "word_count": current_word_count
            })

            chunk_index += 1

            # Keeping last sentences for overlap (context continuity)
            current_sentences = current_sentences[-overlap_sentences:]
            current_word_count = sum(
                len(s.split()) for s in current_sentences 
            )

    # Adding the last chunk which may be smaller than target
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        
        # Only adding words more than 50 for meaningful context!
        if len(chunk_text.split()) > 50:
            chunks.append({
                "content": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
                "word_count": len(chunk_text.split())
            })
    
    logger.info(
        f"Chunked '{source}': {len(sentences)} sentences ->"
        f"{len(chunks)} Chunks"
        f"(avg {sum(c['word_count'] for c in chunks) // max(len(chunks), 1)} words/chunks)"
    )
    return chunks



def process_all_pdf(raw_dir: str = r"data/raw/pdf") -> list[dict]:
    """
    process all pdf's in raw/pdf directories and returns chunks
    """

    pdf_files = [file for file in os.listdir(raw_dir) if file.endswith('.pdf')]

    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}")
        return []
    
    all_chunks = []

    for filename in pdf_files:
        pdf_path = os.path.join(raw_dir, filename)
        logger.info(f"Processing: {filename}")

        try:
            text = extract_text_from_pdf(pdf_path)

            if len(text.split()) < 100:
                logger.error(
                    f"skipping {filename} too little text extracted"
                    f"({len(text.split())} words), Maybe scanned pdf"
                )
                continue

            chunks = chunk_text_by_sentece(
                text = text,
                source = filename,
                target_chunk_words = 300,
                overlap_sentences=2
            )
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Failed to process {filename} : {e}")
            continue

    logger.info(f"total chunks from all the PDF's: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Inspecting our chunks before embedding!
    import json

    print("\n=== Document chunking pipeiline ===")

    all_chunks = process_all_pdf(r"data/raw/pdf")
    
    if not all_chunks:
        print("No chunks produced, check PDF Files")
        sys.exit(1)

    # save full output path for inspection
    output_path = r"data/processed/all_chunks.json"
    os.makedirs(r"data/processed", exist_ok= True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii= False)

    print("\n === Chunking summary ===")
    print(f"Total chunks: {len(all_chunks)}")

    # stats per source
    from collections import Counter
    source_count = Counter([c['source'] for c in all_chunks])
    print("\n Chunks per source:")
    for source, counts in source_count.most_common():
        print(f" {source}: {counts} chunks")
    
    # word count distribution
    word_counts = [c['word_count'] for c in all_chunks]
    print("\nWord count stats")
    print(f"Min: {min(word_counts)}")
    print(f"Max: {max(word_counts)}")
    print(f"Avg: {sum(word_counts)// len(word_counts)}")

    # Preview of first 5 chunks
    preview_path = r"data/processed/chunks_preview.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks[:5], f, indent=2, ensure_ascii=False)

    print(f"\nFull ouput saved to: {output_path}")
    print(f"Preview saved to: {preview_path}")

    print(f"=== SAMPLE CHUNK ===")
    sample = all_chunks[len(all_chunks) // 2] # middle chunk
    print(f"Source: {sample['source']}")
    print(f"Index: {sample['chunk_index']}")
    print(f"Words: {sample['word_count']}")
    print(f"Content:\n{sample['content'][:400]}...")