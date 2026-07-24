"""
RAG Evaluation Script (Ragas-style, no pyarrow dependency)
Evaluates the RAG pipeline against the golden dataset using LLM-as-judge.

Metrics computed:
  - Faithfulness       (generation): Is the answer grounded in the retrieved context?
  - Answer Relevancy   (generation): Does the answer address the question?
  - Context Precision  (retrieval):  Are retrieved chunks relevant to the question?
  - Context Recall     (retrieval):  Do chunks contain all info needed for ground truth?
  - Answer Correctness (end-to-end): Does the answer match the ground truth semantically?

Output: evaluation/results.json + a printed summary table
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GOLDENS_PATH = "evaluation/goldens.json"
RESULTS_PATH = "evaluation/results.json"
RAG_API_URL  = "http://localhost:8000/query"   # Your FastAPI endpoint
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JUDGE_MODEL  = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────
# 1. Query the RAG pipeline
# ─────────────────────────────────────────────
def query_rag(question: str) -> tuple[str, list[str]]:
    """Sends question to the FastAPI RAG endpoint and returns (answer, contexts)."""
    try:
        resp = requests.post(RAG_API_URL, json={"q": question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer   = data.get("answer", "")
        # The API returns 'sources' as a list of document dicts with page_content
        sources  = data.get("sources", [])
        contexts = []
        for s in sources:
            if isinstance(s, dict):
                contexts.append(s.get("page_content", s.get("content", str(s))))
            elif isinstance(s, str):
                contexts.append(s)
        return answer, contexts
    except Exception as e:
        print(f"  ⚠️  RAG API error: {e}")
        return "", []


# ─────────────────────────────────────────────
# 2. LLM-as-Judge helper
# ─────────────────────────────────────────────
def llm_judge(prompt: str) -> float:
    """Calls Groq and extracts a 0.0–1.0 score from the response with rate limit handling."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "EvaluationScript/1.0"
    }
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    # Retry loop for rate limits
    for attempt in range(5):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, headers=headers, timeout=30)
            
            if r.status_code == 429:
                print(f"    ⏳ Rate limited by Groq (Judge). Sleeping for 15s...")
                time.sleep(15)
                continue
                
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            
            import re
            match = re.search(r"\b([01](\.\d+)?|0\.\d+)\b", text)
            return float(match.group(1)) if match else 0.0
            
        except Exception as e:
            print(f"  ⚠️  Judge error: {e}")
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
                print("    ⏳ Forbidden 403 - usually transient Cloudflare block. Retrying...")
                time.sleep(5)
                continue
            return 0.0
    
    return 0.0


# ─────────────────────────────────────────────
# 3. Individual Metric Functions
# ─────────────────────────────────────────────
def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts[:3]) if contexts else "No context retrieved."
    prompt = f"""Score the FAITHFULNESS of an answer from 0.0 to 1.0.
Faithfulness = every claim in the answer is directly supported by the context. No hallucinations.

QUESTION: {question}
CONTEXT: {ctx}
ANSWER: {answer}

Return ONLY a number between 0.0 and 1.0."""
    return llm_judge(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""Score the ANSWER RELEVANCY from 0.0 to 1.0.
Relevancy = the answer directly addresses the question without unnecessary padding.

QUESTION: {question}
ANSWER: {answer}

Return ONLY a number between 0.0 and 1.0."""
    return llm_judge(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    ctx = "\n---\n".join(contexts[:3])
    prompt = f"""Score CONTEXT PRECISION from 0.0 to 1.0.
Precision = the retrieved context chunks are relevant and useful for answering the question.

QUESTION: {question}
RETRIEVED CONTEXT: {ctx}

Return ONLY a number between 0.0 and 1.0."""
    return llm_judge(prompt)


def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    ctx = "\n---\n".join(contexts[:3])
    prompt = f"""Score CONTEXT RECALL from 0.0 to 1.0.
Recall = the retrieved context contains all the information needed to derive the ground truth answer.

GROUND TRUTH: {ground_truth}
RETRIEVED CONTEXT: {ctx}

Return ONLY a number between 0.0 and 1.0."""
    return llm_judge(prompt)


def score_answer_correctness(question: str, answer: str, ground_truth: str) -> float:
    prompt = f"""Score ANSWER CORRECTNESS from 0.0 to 1.0.
Correctness = how semantically similar and factually aligned is the answer to the ground truth?

QUESTION: {question}
GROUND TRUTH: {ground_truth}
ANSWER: {answer}

Return ONLY a number between 0.0 and 1.0."""
    return llm_judge(prompt)


# ─────────────────────────────────────────────
# 4. Main Evaluation Loop
# ─────────────────────────────────────────────
def evaluate():
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        goldens = json.load(f)

    print(f"\n🧪 Starting RAG evaluation on {len(goldens)} golden questions...\n")
    results = []

    for i, gold in enumerate(goldens, 1):
        question     = gold["question"]
        ground_truth = gold["ground_truth"]

        print(f"[{i}/{len(goldens)}] Q: {question[:70]}...")

        # Phase 1: Run the RAG pipeline with a retry loop in case the API hits a Groq 429
        answer, contexts = "", []
        for attempt in range(5):
            answer, contexts = query_rag(question)
            if not answer:
                print("  ⚠️  No answer returned... retrying in 15 seconds (likely Rate Limit)")
                time.sleep(15)
            else:
                break
                
        if not answer:
            print("  ⚠️  Failed to get answer after 5 retries — skipping.")
            continue

        # Phase 2: Score with LLM-as-judge
        faithfulness       = score_faithfulness(question, answer, contexts)
        answer_relevancy   = score_answer_relevancy(question, answer)
        context_precision  = score_context_precision(question, contexts)
        context_recall     = score_context_recall(ground_truth, contexts)
        answer_correctness = score_answer_correctness(question, answer, ground_truth)

        row = {
            "question":           question,
            "ground_truth":       ground_truth,
            "answer":             answer,
            "faithfulness":       faithfulness,
            "answer_relevancy":   answer_relevancy,
            "context_precision":  context_precision,
            "context_recall":     context_recall,
            "answer_correctness": answer_correctness,
        }
        results.append(row)

        print(f"  ✅ Faithfulness={faithfulness:.2f}  Relevancy={answer_relevancy:.2f}  "
              f"Precision={context_precision:.2f}  Recall={context_recall:.2f}  "
              f"Correctness={answer_correctness:.2f}")

        print("  ⏳ Resting for 15s to respect Groq free-tier TPM limits...")
        time.sleep(15)

    # ── Summary ──
    if results:
        metrics = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_correctness"]
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        for m in metrics:
            avg = sum(r[m] for r in results) / len(results)
            bar = "█" * int(avg * 20)
            print(f"  {m:<22} {avg:.3f}  |{bar:<20}|")
        print("="*60)

        # Save results
        os.makedirs("evaluation", exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed results saved to '{RESULTS_PATH}'")


if __name__ == "__main__":
    evaluate()
