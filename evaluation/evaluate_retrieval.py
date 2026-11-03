"""
evaluate_retrieval.py - Rigorous Evaluation Suite for Subject-Wise RAG

Measures:
1. Document-level Top-1, Top-3, and Top-5 Retrieval Accuracy.
2. Subject-Filter Isolation Rate (guaranteeing 0% cross-subject leakage).
3. Citation Provenance Completeness (verifying page or slide numbers on every chunk).
4. Generation of a reproducible evaluation report (retrieval_report.json).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.search import SubjectVectorSearch


def run_evaluation(
    questions_file: str = "evaluation/retrieval_questions.json",
    report_file: str = "evaluation/retrieval_report.json",
    top_k: int = 5
) -> Dict[str, Any]:
    print("=" * 65)
    print("SUBJECT-WISE ACADEMIC RAG - RETRIEVAL EVALUATION")
    print("=" * 65)

    search_engine = SubjectVectorSearch()

    q_path = ROOT_DIR / questions_file
    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total_testable = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    isolation_tests = 0
    isolation_passed = 0
    total_retrieved_chunks = 0
    valid_citations_chunks = 0

    results_detail = []

    for item in questions:
        q_id = item["id"]
        question = item["question"]
        target_subj = item.get("subject_id")
        test_type = item.get("test_type", "retrieval_relevance")

        print(f"\n[{q_id}] Query: '{question}'")
        print(f"      Target Subject: {target_subj} | Type: {test_type}")

        retrieved = search_engine.search(query=question, subject_id=target_subj, top_k=top_k)

        # Subject isolation check
        leakage = False
        forbidden = item.get("forbidden_subjects", [])
        for r in retrieved:
            total_retrieved_chunks += 1
            if r.get("page") is not None or r.get("slide") is not None:
                valid_citations_chunks += 1
            if forbidden and r["subject_id"] in forbidden:
                leakage = True
            elif target_subj and r["subject_id"] != target_subj:
                leakage = True

        if forbidden or test_type == "subject_isolation":
            isolation_tests += 1
            if not leakage:
                isolation_passed += 1
                print("      [OK] Subject Isolation: Passed (0 chunks leaked)")
            else:
                print("      [FAIL] Subject Isolation: Leaked chunks from forbidden subjects!")

        if test_type == "retrieval_relevance":
            total_testable += 1
            expected_kws = item.get("expected_document_keywords", [])

            def matches_keywords(src_name, text):
                combined = (src_name + " " + text).lower()
                return any(kw.lower() in combined for kw in expected_kws)

            top1_match = matches_keywords(retrieved[0]["source_file"], retrieved[0]["text"]) if len(retrieved) >= 1 else False
            top3_match = any(matches_keywords(r["source_file"], r["text"]) for r in retrieved[:3])
            top5_match = any(matches_keywords(r["source_file"], r["text"]) for r in retrieved[:5])

            if top1_match:
                top1_hits += 1
            if top3_match:
                top3_hits += 1
            if top5_match:
                top5_hits += 1

            status = "Top-1 Hit" if top1_match else ("Top-3 Hit" if top3_match else ("Top-5 Hit" if top5_match else "Miss"))
            print(f"      Result: {status} (Top-1: {top1_match}, Top-3: {top3_match}, Top-5: {top5_match})")
            if retrieved:
                top_doc = retrieved[0]
                loc = f"p.{top_doc['page']}" if top_doc.get("page") else f"s.{top_doc.get('slide')}"
                print(f"      Top doc: {top_doc['source_file']} ({loc}) | Dist: {top_doc['distance']:.4f}")

        results_detail.append({
            "id": q_id,
            "question": question,
            "subject": target_subj,
            "retrieved_count": len(retrieved),
            "top_source": retrieved[0]["source_file"] if retrieved else None,
            "top_distance": retrieved[0]["distance"] if retrieved else None
        })

    # Summary metrics
    top1_acc = (top1_hits / total_testable * 100) if total_testable else 0
    top3_acc = (top3_hits / total_testable * 100) if total_testable else 0
    top5_acc = (top5_hits / total_testable * 100) if total_testable else 0
    iso_acc = (isolation_passed / isolation_tests * 100) if isolation_tests else 100
    cit_rate = (valid_citations_chunks / total_retrieved_chunks * 100) if total_retrieved_chunks else 0

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Retrieval Questions Tested: {total_testable}")
    print(f"Top-1 Retrieval Accuracy:        {top1_acc:.1f}% ({top1_hits}/{total_testable})")
    print(f"Top-3 Retrieval Accuracy:        {top3_acc:.1f}% ({top3_hits}/{total_testable})")
    print(f"Top-5 Retrieval Accuracy:        {top5_acc:.1f}% ({top5_hits}/{total_testable})")
    print(f"Subject Filter Isolation:        {iso_acc:.1f}% ({isolation_passed}/{isolation_tests})")
    print(f"Citation Provenance Completeness:{cit_rate:.1f}% ({valid_citations_chunks}/{total_retrieved_chunks} chunks)")
    print("=" * 65)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": top_k,
        "metrics": {
            "top1_accuracy_percent": round(top1_acc, 2),
            "top3_accuracy_percent": round(top3_acc, 2),
            "top5_accuracy_percent": round(top5_acc, 2),
            "subject_isolation_percent": round(iso_acc, 2),
            "citation_provenance_percent": round(cit_rate, 2),
            "total_testable_queries": total_testable,
            "total_chunks_inspected": total_retrieved_chunks
        },
        "details": results_detail
    }

    rep_path = ROOT_DIR / report_file
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Report saved to {rep_path}")

    return report_data


if __name__ == "__main__":
    run_evaluation()
