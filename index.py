#!/usr/bin/env python3
"""
AvMate Regulation Indexer
=========================
Downloads PDFs from Cloudflare R2, extracts text, chunks it, generates embeddings,
and stores in Chroma vector database for fast semantic search.

Usage:
    python3 index.py

Requirements:
    pip install requests pdfplumber chromadb sentence-transformers
"""

import os
import requests
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import re
import pytesseract
from pdf2image import convert_from_path

# Set Tesseract path (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# R2 bucket URL
R2_BASE_URL = "https://pub-a32237578ade418f9375e48bb3f1982a.r2.dev"

# List of PDF files (from the regulation folder)
PDF_FILES = [
    "AIP complete_19MAR2026.pdf",
    "CAA.pdf",
    "CAO 48.1.pdf",
    "CAR VOL01.pdf",
    "CAR VOL02.pdf",
    "CASR VOL01.pdf",
    "CASR VOL02.pdf",
    "CASR VOL03.pdf",
    "CASR VOL04.pdf",
    "CASR VOL05.pdf",
    "flight-crew-licensing-manual.pdf",
    "flight-examiner-handbook.pdf",
    # MOS files (add more if needed, or automate listing)
    "MOS/F2015C00263.pdf",
    "MOS/F2016L01762.pdf",
    "MOS/F2017C01160REC01.pdf",
    "MOS/F2021L01654.pdf",
    "MOS/F2021L01655.pdf",
    "MOS/F2022C01244.pdf",
    "MOS/F2023C00278.pdf",
    "MOS/F2023L01107.pdf",
    "MOS/F2024C00025.pdf",
    "MOS/F2024C00332.pdf",
    "MOS/F2024C00333.pdf",
    "MOS/F2024C01027VOL01.pdf",
    "MOS/F2024C01027VOL02.pdf",
    "MOS/F2024C01027VOL03.pdf",
    "MOS/F2024C01027VOL04.pdf",
    "MOS/F2024C01027VOL05.pdf",
    "MOS/F2024L01451.pdf",
    "MOS/F2024N00789.pdf",
    "MOS/F2025C00050VOL01REC01.pdf",
    "MOS/F2025C00050VOL02REC01.pdf",
    "MOS/F2025C00050VOL03REC01.pdf",
    "MOS/F2025C00050VOL04REC01.pdf",
    "MOS/F2025C00187.pdf",
    "MOS/F2025C00701.pdf",
    "MOS/F2025C00829.pdf",
    "MOS/F2025C01005.pdf",
    "MOS/F2025C01184.pdf",
    "MOS/F2025C01196.pdf",
    "MOS/F2025C01225.pdf",
    "MOS/F2025L01465.pdf",
]

def download_pdf(url, filename):
    """Download PDF from URL to a temporary file."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        print(f"Failed to download {url}")
        return None

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
    
    if not text.strip():  # If no text, use OCR
        print(f"No text found, using OCR for {pdf_path}")
        try:
            images = convert_from_path(pdf_path)
            for image in images:
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
        except Exception as e:
            print(f"OCR failed for {pdf_path}: {e}")
    
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
        url = f"{R2_BASE_URL}/{pdf_file.replace(' ', '%20').replace('/', '%2F')}"
        temp_file = f"temp_{os.path.basename(pdf_file)}"
        print(f"Processing {pdf_file}...")
        print(f"Downloading {url}")
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        if response.status_code != 200:
            print(f"Failed to download {url}: {response.text}")
            continue
        with open(temp_file, 'wb') as f:
            f.write(response.content)
            text = extract_text_from_pdf(temp_file)
            chunks = chunk_text(text)
            embeddings = model.encode(chunks)
            # Add to Chroma
            ids = [f"{pdf_file}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_file, "chunk_id": i} for i in range(len(chunks))]
            collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            os.remove(temp_file)
        else:
            print(f"Skipping {pdf_file}")

    print("Indexing complete.")

if __name__ == "__main__":
    main()