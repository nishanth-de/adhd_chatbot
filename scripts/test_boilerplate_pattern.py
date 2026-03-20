from chunk_documents import extract_text_from_pdf

nice_pdf = r"data/raw/pdf/NHIM_adhd-overview_2024.pdf"  # replace with actual name
text = extract_text_from_pdf(nice_pdf)

# Count remaining occurrences of the footer
footer_count = text.count("NICE 2025")
print(f"Remaining NICE footer occurrences: {footer_count}")
print(f"Total words remaining: {len(text.split())}")