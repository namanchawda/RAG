"""Evaluate the baseline SEC RAG pipeline against the evaluation set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.generation.llm_client import generate_answer
from app.generation.rag_pipeline import answer_question
from app.ingestion import store


ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results_baseline.json"
PARTIAL_RESULTS_PATH = Path(__file__).resolve().parent / "results_partial_run.json"
CATEGORY_ORDER = [
    "direct",
    "semantic_paraphrase",
    "numeric_lookup",
    "cross_reference",
    "out_of_scope",
]


def judge_answer(question: str, expected_answer: str, actual_answer: str) -> str:
    """Use Gemini as an LLM judge to decide whether the answer passes."""
    prompt = (
        "You are grading a retrieval-augmented generation answer. "
        "Respond with ONLY PASS or FAIL. "
        "PASS means the actual answer contains the key facts from the expected answer. "
        "For out-of-scope questions, PASS means the system correctly says it does not have enough information in context. "
        "Do not explain your reasoning.\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected_answer}\n\n"
        f"Actual answer: {actual_answer}\n\n"
        "Return only PASS or FAIL."
    )

    verdict = generate_answer(prompt).strip().upper()
    if verdict not in {"PASS", "FAIL"}:
        verdict = "FAIL"
    return verdict


def compute_category_summary(results: list[dict]) -> list[dict]:
    """Return pass counts and rates grouped by evaluation category."""
    summary: list[dict] = []

    for category in CATEGORY_ORDER:
        matches = [result for result in results if result["category"] == category]
        total = len(matches)
        passed = sum(1 for result in matches if result["verdict"] == "PASS")
        rate = (passed / total * 100) if total else 0.0
        summary.append(
            {
                "category": category,
                "passed": passed,
                "total": total,
                "pass_rate": round(rate, 1),
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    """Parse optional evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the SEC RAG baseline against the eval set.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many questions to run from eval_set.json.",
    )
    return parser.parse_args()


def run_evaluation(limit: int | None = None) -> list[dict]:
    """Run the baseline evaluation and save the results JSON."""
    store._ensure_initialized()
    eval_items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if limit is not None:
        eval_items = eval_items[:limit]
        output_path = PARTIAL_RESULTS_PATH
        run_label = f"limited run ({len(eval_items)} questions)"
    else:
        output_path = RESULTS_PATH
        run_label = "full run"

    results: list[dict] = []

    for index, item in enumerate(eval_items, start=1):
        question_id = item["id"]
        category = item.get("category", "unknown")
        question = item["question"]
        expected_answer = item["expected_answer"]
        source_file = item.get("source_file")

        print(f"Running {question_id}/{len(eval_items)}...")

        try:
            response = answer_question(question, source_file=source_file)
            actual_answer = response.get("answer", "")
            sources = response.get("sources", [])
        except Exception as exc:  # pragma: no cover - runtime guard for API / retrieval issues
            actual_answer = f"ERROR: {exc}"
            sources = []

        verdict = judge_answer(question, expected_answer, actual_answer)

        results.append(
            {
                "id": question_id,
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "verdict": verdict,
                "sources": sources,
            }
        )

        if index < len(eval_items):
            time.sleep(1.2)

    total = len(results)
    passed = sum(1 for result in results if result["verdict"] == "PASS")
    overall_rate = (passed / total * 100) if total else 0.0

    print(f"\nOverall: {passed}/{total} passed = {overall_rate:.1f}%")
    print(f"Run type: {run_label}")
    print("Pass rate by category:")
    for row in compute_category_summary(results):
        print(
            f"- {row['category']}: {row['passed']}/{row['total']} passed = {row['pass_rate']:.1f}%"
        )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {output_path}")
    return results


if __name__ == "__main__":
    args = parse_args()
    store._ensure_initialized()
    run_evaluation(limit=args.limit)
