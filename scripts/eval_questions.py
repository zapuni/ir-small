#!/usr/bin/env python3
"""
Evaluate the RAG pipeline against a generated question set (in-process).

Mirrors the /ask handler exactly: parse MCQ -> option-aware hybrid retrieve ->
trim context -> LLM (with deadline) -> embedding fallback. Scores answers
against the ground-truth 'answer' field and reports accuracy by difficulty/type,
latency, and the list of wrong questions.

Usage:
    python scripts/eval_questions.py                          # default file, all Q
    python scripts/eval_questions.py --file contens_notifications --limit 50
    python scripts/eval_questions.py --model intfloat/multilingual-e5-base
    python scripts/eval_questions.py --no-llm                 # fallback (retrieval) only
    python scripts/eval_questions.py --all                    # every question file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

QDIR = PROJECT_ROOT / "tmp" / "scrapePortalData" / "question"
DOCDIR = PROJECT_ROOT / "tmp" / "scrapePortalData" / "merge-content"


def evaluate_file(stem: str, limit: int | None, use_llm: bool) -> dict:
    import config
    import retriever
    from embedder import warm_up
    from llm import fallback_answer, get_client, trim_contexts
    from textutils import parse_mc_question

    doc_path = DOCDIR / f"{stem}.txt"
    q_path = QDIR / f"{stem}.json"
    text = doc_path.read_text(encoding="utf-8")
    questions = json.loads(q_path.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]

    print(f"\n{'='*72}\nFILE: {stem}  | model={config.EMBED_MODEL_NAME} | "
          f"questions={len(questions)} | llm={'on' if use_llm else 'off (fallback only)'}")
    print("="*72)

    warm_up()
    t_idx = time.time()
    n_chunks = retriever.INDEX.build(text, doc_id=stem, use_cache=True)
    print(f"Indexed {n_chunks} chunks in {time.time()-t_idx:.1f}s")

    client = None
    if use_llm:
        try:
            client = get_client()
        except Exception as exc:
            print(f"LLM init failed ({exc!r}); running fallback-only.")
            use_llm = False

    correct = 0
    ans_in_ctx = 0
    by_diff: dict[str, list[int]] = {}
    by_type: dict[str, list[int]] = {}
    wrong: list[dict] = []
    t0 = time.time()

    import re as _re

    def _norm(s: str) -> str:
        return _re.sub(r"\s+", " ", s).strip().lower()

    for i, q in enumerate(questions, 1):
        mcq = parse_mc_question(q["question"])
        contexts = trim_contexts(
            retriever.INDEX.search_mcq(mcq.stem or mcq.raw, mcq.options, top_k=config.TOP_K_CONTEXT)
        )

        # Retrieval recall: is the gold answer value present in the context?
        ctx_joined = _norm(" ".join(contexts))
        has_ans = _norm(q.get("answer_text", "")) in ctx_joined
        ans_in_ctx += has_ans

        letter = None
        if use_llm and client is not None:
            try:
                letter = client.answer(mcq, contexts, deadline=time.monotonic() + config.ASK_DEADLINE)
            except Exception:
                letter = None
        if letter not in ("A", "B", "C", "D"):
            letter = fallback_answer(mcq, contexts)

        gold = q["answer"]
        ok = letter == gold
        correct += ok
        d, t = q.get("difficulty", "?"), q.get("type", "?")
        by_diff.setdefault(d, []).append(ok)
        by_type.setdefault(t, []).append(ok)
        if not ok:
            wrong.append({"id": q["id"], "diff": d, "type": t, "gold": gold,
                          "got": letter, "ans": q.get("answer_text", ""),
                          "recall": has_ans})
        if i % 20 == 0:
            print(f"  ...{i}/{len(questions)}  running acc={correct/i:.1%} recall={ans_in_ctx/i:.1%}")

    elapsed = time.time() - t0
    acc = correct / len(questions) if questions else 0.0
    recall = ans_in_ctx / len(questions) if questions else 0.0

    print(f"\nACCURACY: {correct}/{len(questions)} = {acc:.1%}   "
          f"({elapsed:.0f}s total, {elapsed/max(1,len(questions)):.2f}s/q)")
    print(f"RETRIEVAL RECALL (answer value in context): {ans_in_ctx}/{len(questions)} = {recall:.1%}")
    print("By difficulty:", {d: f"{sum(v)}/{len(v)}" for d, v in sorted(by_diff.items())})
    print("By type:      ", {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_type.items())})
    if wrong:
        print(f"\nWRONG ({len(wrong)}):")
        for w in wrong[:40]:
            print(f"  #{w['id']:>3} [{w['diff']}/{w['type']}] gold={w['gold']} got={w['got']} "
                  f"recall={'Y' if w.get('recall') else 'N'} | {w['ans']}")
    return {"stem": stem, "acc": acc, "correct": correct, "total": len(questions), "wrong": wrong}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="contens_notifications", help="question file stem (no .json)")
    ap.add_argument("--all", action="store_true", help="evaluate all question files")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    ap.add_argument("--model", default=None, help="override EMBED_MODEL_NAME")
    ap.add_argument("--no-llm", action="store_true", help="fallback (retrieval) only, no LLM")
    args = ap.parse_args()

    if args.model:
        os.environ["EMBED_MODEL_NAME"] = args.model

    stems = ([p.stem for p in sorted(QDIR.glob("*.json"))] if args.all else [args.file])

    results = []
    for stem in stems:
        results.append(evaluate_file(stem, args.limit, use_llm=not args.no_llm))

    if len(results) > 1:
        print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
        tot_c = sum(r["correct"] for r in results)
        tot_n = sum(r["total"] for r in results)
        for r in results:
            print(f"  {r['stem']:<32} {r['correct']}/{r['total']} = {r['acc']:.1%}")
        print(f"  {'OVERALL':<32} {tot_c}/{tot_n} = {tot_c/max(1,tot_n):.1%}")


if __name__ == "__main__":
    main()
