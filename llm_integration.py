import os
import json
import faiss
import torch
import re
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Constants
FAISS_DIR = "Data/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index_flatl2.index")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

def load_models():
    try:
        print("🔄 Loading embedding model...")
        embedding_model = SentenceTransformer("BAAI/bge-large-en", device="cpu")

        print("🔧 Loading TinyLlama model...")
        llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float32)
        model.eval()

        return embedding_model, tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

model_embedding, tokenizer, model_llm = load_models()

def extract_course_code(query: str, metadata: List[Dict]) -> Optional[str]:
    code_match = re.search(r'\b\d{2}[A-Z]{3}\d{3}[A-Z]?\b', query)
    if code_match:
        return code_match.group(0)

    query_embedding = model_embedding.encode([query], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatL2(query_embedding.shape[1])
    index.add(np.array([m['embedding'] for m in metadata if 'embedding' in m]))

    distances, indices = index.search(query_embedding, 3)
    for idx in indices[0]:
        if idx < len(metadata):
            content = metadata[idx]['content']
            code_in_content = re.search(r'\b\d{2}[A-Z]{3}\d{3}[A-Z]?\b', content)
            if code_in_content:
                return code_in_content.group(0)
    return None

def extract_course_specific_content(content: str, course_code: str) -> str:
    # Attempt to extract CLR content with a more flexible pattern
    pattern = rf"(Course Learning Rationale|CLR).*?{re.escape(course_code)}.*?(?=(Course Outcomes|CO|Unit|$))"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0)

    # Fallback pattern for CLR, looking for a more general block of CLR-related text
    fallback = re.search(r"(Course Learning Rationale|CLR).*?(?=\n{2,}|$)", content, re.DOTALL | re.IGNORECASE)
    if fallback:
        return fallback.group(0)

    return ""

def clean_table_content(text: str) -> str:
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r'(CLR|CO|Unit|T)-\d+', lambda m: m.group().replace(' ', ''), text)
    text = re.sub(r'\b\d{2}[A-Z]{3}\d{3}[A-Z]?\b', lambda m: f"COURSE:{m.group()}", text)
    text = re.sub(r'[ \t]+', ' ', text)  # Preserve newlines
    return text.strip()

def format_response(response: str, course_code: str) -> str:
    response = re.sub(r'(Program Outcomes|Program Specific Outcomes).*$', '', response, flags=re.DOTALL)
    if course_code and f"COURSE:{course_code}" not in response:
        response = f"For COURSE:{course_code}:\n{response}"
    return response.replace("COURSE:", "").strip()

def get_llm_response(query: str, context: str, course_code: str) -> str:
    prompt = (
        f"<|system|>\nYou are a helpful university course assistant. Only use the information from COURSE:{course_code}. "
        f"Respond concisely and clearly. If Course Learning Rationale (CLR) is present, list each point in a numbered format.\n\n"
        f"Current Context:\n{context[:1500]}\n\n"
        f"<|user|>\nWhat is the Course Learning Rationale (CLR) for COURSE:{course_code}?\n"
        f"<|assistant|>\nAnswer in numbered points:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)

    with torch.no_grad():
        outputs = model_llm.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=400,   # ← increased from 250
            do_sample=False,
            num_beams=3,
            temperature=0.7
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return format_response(full_response.split("Answer in numbered points:")[-1], course_code)


def search_and_generate(query: str) -> str:
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        course_code = extract_course_code(query, metadata)
        if not course_code:
            return "Please specify a course code (e.g., 21CSE313P) in your question."

        print(f"[DEBUG] Extracted course code: {course_code}")

        query_embedding = model_embedding.encode(
            [f"{course_code} {query}"],
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = index.search(query_embedding, 5)
        context_chunks = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(metadata) and dist < 1.0:
                content = metadata[idx]['content']
                print(f"[DEBUG] Checking content from Page {metadata[idx]['page']}")

                if course_code in content:
                    print(f"[DEBUG] Course code {course_code} found in content.")
                    cleaned = clean_table_content(content)
                    course_content = extract_course_specific_content(cleaned, course_code)

                    if course_content:
                        print(f"[DEBUG] CLR content extracted for {course_code}")
                        context_chunks.append(f"From Page {metadata[idx]['page']}:\n{course_content}")
                    else:
                        print(f"[DEBUG] No CLR section found for {course_code} in this chunk.")
                else:
                    print(f"[DEBUG] Course code {course_code} NOT found in chunk.")

        print(f"[DEBUG] Context chunks found: {len(context_chunks)}")

        if not context_chunks:
            return f"No relevant content found for course {course_code}."

        return get_llm_response(query, "\n\n".join(context_chunks), course_code)

    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("=== University Course Q&A ===")
    print("Ask about any course (include course code like 21CSE313P)\nType 'exit' to quit\n")

    while True:
        try:
            user_input = input("👤 Question: ").strip()
            if user_input.lower() in ('exit', 'quit'):
                break
            response = search_and_generate(user_input)
            print(f"\n🤖 Response:\n{response}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
