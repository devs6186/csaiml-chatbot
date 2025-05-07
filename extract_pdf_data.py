import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import os
import json
import re

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Define base directory and output folder
BASE_DIR = "Data"
PDF_FILES = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".pdf")]
OUTPUT_DIR = os.path.join(BASE_DIR, "Extracted")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_context(context):
    """
    Cleans the extracted context, removes page numbers, and keeps course codes.
    """
    # Remove page numbers (e.g., 573) but keep course codes
    context = re.sub(r'\b\d{3,}\b', '', context)  # This will remove numbers like 573, etc.

    # Remove ellipses (…) and excessive spaces
    context = context.replace('…………………………………………………………', ' ').replace('…', ' ')
    context = ' '.join(context.split())  # Remove multiple spaces and newlines
    
    return context


def extract_data_per_page(pdf_path):
    # Unified extraction
    data = []
    doc = fitz.open(pdf_path)
    plumber_pdf = pdfplumber.open(pdf_path)

    for i in range(len(doc)):
        page_number = i + 1
        text = doc[i].get_text("text").strip()
        images = doc[i].get_images(full=True)
        image_texts = []

        # OCR extraction from images
        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(pdf_path).replace('.pdf','')}_p{page_number}_img{img_index+1}.{image_ext}")

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                img_pil = Image.open(image_path)
                ocr_text = pytesseract.image_to_string(img_pil)
                if ocr_text.strip():
                    image_texts.append(ocr_text.strip())

            except Exception as e:
                print(f"❌ OCR failed on page {page_number} image {img_index+1}: {e}")

        # Extract tables
        try:
            tables_raw = plumber_pdf.pages[i].extract_tables({
                "vertical_strategy": "text", 
                "horizontal_strategy": "text",
                "intersection_y_tolerance": 10
            })
            
            # Process tables (remove empty cells and structure the data)
            tables = []
            for table in tables_raw:
                cleaned_table = []
                for row in table:
                    cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                    if any(cleaned_row):  # Only keep rows with content
                        cleaned_table.append(cleaned_row)
                if cleaned_table:  # Only keep tables with content
                    tables.append(cleaned_table)
        except Exception as e:
            print(f"❌ Table extraction failed on page {page_number}: {e}")
            tables = []

        # Clean the text context (removes unnecessary page numbers)
        cleaned_text = clean_context(text)

        # Improved metadata for course content
        data.append({
            "page_number": page_number,
            "text": cleaned_text,
            "image_texts": image_texts,
            "tables": tables,
            "is_structured_content": bool(tables)  # Flag for course tables
        })

    doc.close()
    plumber_pdf.close()
    return data


def save_json(data, filename):
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    for pdf in PDF_FILES:
        print(f"\n📄 Processing: {pdf}")
        all_data = extract_data_per_page(pdf)
        base_name = os.path.splitext(os.path.basename(pdf))[0]
        save_json(all_data, f"{base_name}_full_extracted.json")

    print(f"\n✅ Done! Extracted content saved in '{OUTPUT_DIR}' folder.")
