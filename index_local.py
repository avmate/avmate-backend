#!/usr/bin/env python3
"""
AvMate Local Indexer
====================
Reads PDFs directly from local disk, extracts text, chunks it,
generates embeddings, and stores in ChromaDB.

Usage:
    python index_local.py
"""

import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

LOCAL_PDF_DIR = r"c:\Users\Taylz\Documents\AvMate App\Regulation files"

PDF_FILES = [
    r"AIP\AIP_complete_19MAR2026.pdf",
    r"CAA\CAA.pdf",
    r"CAO\CAO_48.1.pdf",
    r"CAR\CAR_VOL01.pdf",
    r"CAR\CAR_VOL02.pdf",
    r"CASR\CASR_VOL01.pdf",
    r"CASR\CASR_VOL02.pdf",
    r"CASR\CASR_VOL03.pdf",
    r"CASR\CASR_VOL04.pdf",
    r"CASR\CASR_VOL05.pdf",
    r"Manuals\flight_crew_licensing_manual.pdf",
    r"Manuals\flight_examiner_handbook.pdf",
    r"MOS\Part_61_Manual_of_Standards_(MOS)_1.pdf",
    r"MOS\Part_61_Manual_of_Standards_(MOS)_2.pdf",
    r"MOS\Part_61_Manual_of_Standards_(MOS)_3.pdf",
    r"MOS\Part_61_Manual_of_Standards_(MOS)_4.pdf",
    r"MOS\Part_91_(General_Operating_and_Flight_Rules)_Manual_of_Standards_2020.pdf",
    r"MOS\Part_121_(Australian_Air_transport_Operations-Larger_Aeroplanes)_Manual_of_Standards_2025.pdf",
    r"MOS\Part_135_(Australian_Air_Transport_Operations-Smaller_Aeroplanes)_Manual_of_Standards_2020.pdf",
    r"MOS\Part_138_(Aerial_Work_Operations)_Manual_of_Standards_2020_(as_amended).pdf",
    r"MOS\Part_139_(Aerodromes)_Manual_of_Standards_2019_(as_amended).pdf",
    r"MOS\Part_145_Manual_of_Standards_(MOS)_(as amended).pdf",
]

def extract_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  Failed to extract text: {e}")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def main():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="avmate_regulations")
    print(f"Starting index. Current collection count: {collection.count()}")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for rel_path in PDF_FILES:
        full_path = os.path.join(LOCAL_PDF_DIR, rel_path)
        source_key = rel_path.replace("\\", "/")

        if not os.path.exists(full_path):
            print(f"NOT FOUND: {full_path}")
            continue

        print(f"Processing {source_key}...")
        text = extract_text(full_path)
        if not text.strip():
            print(f"  No text extracted, skipping.")
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks)
        ids = [f"{source_key}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_key, "chunk_id": i} for i in range(len(chunks))]

        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  Added {len(chunks)} chunks.")

    print(f"\nIndexing complete. Total chunks: {collection.count()}")

if __name__ == "__main__":
    main()
