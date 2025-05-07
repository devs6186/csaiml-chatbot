import os
import json
import re
from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')

# Constants
EXTRACTED_DIR = "Data/Extracted"
OUTPUT_FILE = "Data/cleaned_chunks.json"

# Target roughly ~1500 characters (approx 350–400 tokens)
MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 200  # To maintain context across chunks


def load_all_json():
    """Loads all extracted JSON data from the 'Extracted' folder."""
    data = []
    for file in os.listdir(EXTRACTED_DIR):
        if file.endswith("_full_extracted.json"):
            with open(os.path.join(EXTRACTED_DIR, file), "r", encoding="utf-8") as f:
                json_data = json.load(f)
                base_name = Path(file).stem.replace("_full_extracted", "")
                for entry in json_data:
                    data.append({
                        "source": base_name,
                        "page": entry["page_number"],
                        "text": entry.get("text", ""),
                        "image_texts": entry.get("image_texts", []),
                        "tables": entry.get("tables", [])
                    })
    return data


def clean_text(text):
    """Cleans the text by removing extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def flatten_table(table):
    """Flattens table data into a text format (row-wise)."""
    return "\n".join([ 
        " | ".join([str(cell).strip() if cell is not None else "" for cell in row]) 
        for row in table if any(cell and str(cell).strip() for cell in row)
    ])


def hierarchical_chunk(text, max_chars=MAX_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Chunks the text hierarchically, splitting it into smaller parts with some overlap."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap for RAG continuity
    overlapped_chunks = []
    for i in range(len(chunks)):
        prev_overlap = chunks[i - 1][-overlap:] if i > 0 else ""
        overlapped_chunks.append((prev_overlap + " " + chunks[i]).strip())

    return overlapped_chunks


def process_entries(raw_data):
    """Processes raw data into clean, chunked content, handling both structured and unstructured data."""
    final_chunks = []

    for entry in tqdm(raw_data, desc="🔧 Cleaning and Chunking"):
        # Initialize the final text content
        full_text = clean_text(entry["text"])

        # Add image text content (if any)
        for img_text in entry["image_texts"]:
            full_text += "\n" + clean_text(img_text)

        # Process structured course content (tables)
        if entry.get("is_structured_content"):
            for table in entry["tables"]:
                for row_idx, row in enumerate(table):
                    if any(cell.strip() for cell in row):  # Only process rows with content
                        chunk_content = " | ".join(row)
                        chunk_id = f"{entry['source']}_p{entry['page']}_t{row_idx+1}"
                        final_chunks.append({
                            "chunk_id": chunk_id,
                            "source": entry["source"],
                            "page": entry["page"],
                            "content": chunk_content,
                            "is_structured": True
                        })

        # Process unstructured content
        if not entry.get("is_structured_content"):
            # Flatten tables and append to full text
            for table in entry["tables"]:
                full_text += "\n" + flatten_table(table)

            if full_text.strip():  # Ensure there is content to chunk
                chunks = hierarchical_chunk(full_text)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{entry['source']}_p{entry['page']}_c{idx+1}"
                    final_chunks.append({
                        "chunk_id": chunk_id,
                        "source": entry["source"],
                        "page": entry["page"],
                        "content": chunk,
                        "is_structured": False
                    })

    return final_chunks


if __name__ == "__main__":
    raw_data = load_all_json()
    cleaned_chunks = process_entries(raw_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(cleaned_chunks, out, indent=2, ensure_ascii=False)

    print(f"\n✅ Cleaned & chunked data saved to: {OUTPUT_FILE}")
