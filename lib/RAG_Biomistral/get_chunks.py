# Extract chunks from several pdf files
import fitz  # PyMuPDF

def extract_fulltext_from_pdf(book_pdf):
    doc = fitz.open(book_pdf)
    full_text = ""

    # Extract text from each page and concatenate
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text += text.strip() + " "

    return full_text
def extract_fulltext_from_pdf2(book_pdf):
    doc = fitz.open(book_pdf)
    full_text = ""

    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text += text.strip() + " "

    return full_text

def split_text_into_chunks(text, max_tokens, overlap):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap  # Slide window

    return chunks

