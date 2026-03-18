# PDF Text Extractor
# Writing a Function that reads a PDF and extracts clean text.

import os
import sys
import logging
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

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
        text = re.sub(r"-\n", "", text)

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

    return full_text.strip()

if __name__ == "__main__":
    # Text extraction of the first pdf in our data/raw folder.
    raw_dir = r"data/raw/pdf~"
    pdf_files = [files for files in os.listdir(raw_dir) if files.endswith('.pdf')]

    if not pdf_files:
        print("No PDFs found in data/raw/. Add your PDF files first.")
        sys.exit(1)

    test_pdf = os.path.join(raw_dir, pdf_files[0])
    print(f"Testing extraction on: {test_pdf}\n")

    text = extract_text_from_pdf(test_pdf)
    print(f"Total characters extracted: {len(text)}")
    print(f"Estimated words: {len(text.split())}")
    print(f"\nFirst 500 characters:")
    print("-" * 40)
    print(text[:500])
    print("-" * 40)
    print(f"\nLast 200 characters:")
    print(text[-200:])
