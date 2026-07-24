"""
Ragas Testset Generator (ragas >= 0.2, no pyarrow/HF datasets dependency)
Generates a golden dataset using direct LLM calls via Groq.
Output: evaluation/goldens.json
"""

import os
import json
import glob
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "data"
OUTPUT_DIR = "evaluation"
OUTPUT_FILE = "goldens.json"


def generate_goldens():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_groq import ChatGroq
    from langchain_core.documents import Document

    print("🤖 Initializing Ragas-style Golden Generator (Groq LLM powered)...")

    # Use 70b for high quality ground truths
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)

    # Load a single PDF to stay within rate limits
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))[:1]
    if not pdf_files:
        print("❌ No PDFs found in data/ directory.")
        return

    pdf_path = pdf_files[0]
    print(f"📄 Loading: {os.path.basename(pdf_path)}")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Use the first 10 pages (to avoid token limits)
    pages = pages[:10]
    full_text = "\n\n".join([p.page_content for p in pages if p.page_content.strip()])
    print(f"   Loaded {len(pages)} pages ({len(full_text)} chars)")

    # Build the prompt to generate QA pairs
    prompt = f"""You are an expert at creating evaluation datasets for RAG systems.

Given the following document text, generate exactly 5 question-answer pairs.
Each pair must be:
- A specific, factual question that can only be answered from the text below
- A precise, accurate ground truth answer based strictly on the text

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "question": "...",
    "ground_truth": "...",
    "context": "...the exact sentence(s) from the document that support the answer..."
  }}
]

DOCUMENT TEXT:
{full_text[:8000]}

Return only the JSON array, no other text."""

    print("⚙️  Generating 5 golden QA pairs via Groq 70B... (may take ~30 seconds)")

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        goldens = json.loads(raw)

        # Save output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(goldens, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Successfully saved {len(goldens)} golden QA pairs to '{output_path}'")
        print("\nSample Golden:")
        print(f"  Q: {goldens[0]['question']}")
        print(f"  A: {goldens[0]['ground_truth'][:100]}...")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print("Raw response from LLM:")
        print(raw[:500])
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    generate_goldens()
