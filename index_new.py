#!/usr/bin/env python3
"""
AvMate Regulation Indexer
=========================
Downloads PDFs from Cloudflare R2, extracts text, chunks it, generates embeddings,
and stores in Chroma vector database for fast semantic search.

Usage:
    python3 index.py

Requirements:
    pip install requests pdfplumber chromadb sentence-transformers pytesseract pdf2image
"""

import os
import requests
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from urllib.parse import quote

# R2 bucket URL
R2_BASE_URL = "https://pub-a32237578ade418f9375e48bb3f1982a.r2.dev"

# List of PDF files (relative paths within the R2 bucket)
# Prefix: "Regulation files/" — folder name has a space, encoded as %20 in URLs
PDF_FILES = [
    "Regulation files/AIP/AIP_complete_19MAR2026.pdf",
    "Regulation files/CAA/CAA.pdf",
    "Regulation files/CAO/CAO_48.1.pdf",
    "Regulation files/CAR/CAR_VOL01.pdf",
    "Regulation files/CAR/CAR_VOL02.pdf",
    "Regulation files/CASR/CASR_VOL01.pdf",
    "Regulation files/CASR/CASR_VOL02.pdf",
    "Regulation files/CASR/CASR_VOL03.pdf",
    "Regulation files/CASR/CASR_VOL04.pdf",
    "Regulation files/CASR/CASR_VOL05.pdf",
    "Regulation files/Manuals/flight_crew_licensing_manual.pdf",
    "Regulation files/Manuals/flight_examiner_handbook.pdf",
]

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF, using OCR if necessary."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"pdfplumber failed for {pdf_path}: {e}")
    
    if not text.strip():  # If no text, try OCR (only if system deps available)
        print(f"No text found, attempting OCR for {pdf_path}")
        try:
            import pytesseract
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path)
            for image in images:
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
        except Exception as e:
            print(f"OCR not available or failed for {pdf_path}: {e}")
    
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Chunk text into smaller pieces."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def main():
    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="avmate_regulations")

    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    for pdf_file in PDF_FILES:
        url = f"{R2_BASE_URL}/{quote(pdf_file, safe='/')}"
        temp_file = f"temp_{os.path.basename(pdf_file)}"
        print(f"Processing {pdf_file}...")
        print(f"Downloading {url}")
        try:
            response = requests.get(url, timeout=60)
            print(f"Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Failed to download {url}")
                continue
            with open(temp_file, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Download error for {pdf_file}: {e}")
            continue
        
        text = extract_text_from_pdf(temp_file)
        if not text.strip():
            print(f"No text extracted from {pdf_file}")
            os.remove(temp_file)
            continue
        
        chunks = chunk_text(text)
        if not chunks:
            print(f"No chunks from {pdf_file}")
            os.remove(temp_file)
            continue
        
        embeddings = model.encode(chunks)
        ids = [f"{pdf_file}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_file, "chunk_id": i} for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        os.remove(temp_file)
        print(f"Added {len(chunks)} chunks from {pdf_file}")

    print("Indexing complete.")

if __name__ == "__main__":
    main()