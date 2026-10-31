"""Politika indeksini kurar ve retrieval kalitesini ölçer (hit@1, hit@3, MRR).

Kullanım:
    python scripts/build_policy_index.py                    # e5-small (varsayılan)
    python scripts/build_policy_index.py --compare          # e5 ve trigram yedeğini karşılaştır

Çıktı: index/policy_index.npz (deploy'a dosya olarak girer) ve eval/results/retrieval.csv
[Ref: ShopOps §8 Adım 4 "search_policies('hasarlı ürün') doğru dokümanı ilk 3'te getiriyor"]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.knowledge.policy_search import (PolicyIndex, TrigramEmbedder, default_embedder,  # noqa: E402
                                         load_chunks)

EVAL_FILE = config.ROOT / "eval" / "policy_retrieval.jsonl"
RESULTS = config.ROOT / "eval" / "results" / "retrieval.csv"


def evaluate_index(idx: PolicyIndex, queries: list[dict]) -> dict:
    h1 = h3 = 0
    rr = 0.0
    misses = []
    for q in queries:
        docs = [h.doc for h in idx.search(q["query"], k=len(idx.chunks))]
        # Aynı dokümanın birden çok bölümü olabilir: doküman sırası ilk görünüşe göre
        ranked = list(dict.fromkeys(docs))
        rank = ranked.index(q["expected_doc"]) + 1 if q["expected_doc"] in ranked else None
        top3 = list(dict.fromkeys(h.doc for h in idx.search(q["query"], k=3)))
        h1 += rank == 1
        h3 += q["expected_doc"] in top3
        rr += 1 / rank if rank else 0
        if q["expected_doc"] not in top3:
            misses.append((q["query"], q["expected_doc"], top3))
    n = len(queries)
    return {"embedder": idx.embedder.name, "n": n, "hit@1": round(h1 / n, 3),
            "hit@3": round(h3 / n, 3), "mrr": round(rr / n, 3), "misses": misses}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--embedder", default=config.EMBED_MODEL)
    p.add_argument("--compare", action="store_true", help="Trigram yedeğini de ölç")
    args = p.parse_args(argv)

    chunks = load_chunks(config.POLICIES_DIR)
    queries = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"{len(chunks)} chunk, {len(queries)} sorgu. Embedder: {args.embedder} (ilk sefer modeli indirir)")
    main_idx = PolicyIndex.build(chunks, default_embedder(args.embedder))
    main_idx.save(config.POLICY_INDEX_PATH)
    print(f"İndeks kaydedildi: {config.POLICY_INDEX_PATH}")

    indexes = [main_idx]
    if args.compare and args.embedder != TrigramEmbedder.name:
        indexes.append(PolicyIndex.build(chunks, TrigramEmbedder()))

    rows = []
    for idx in indexes:
        r = evaluate_index(idx, queries)
        rows.append(r)
        print(f"\n{r['embedder']}: hit@1={r['hit@1']:.3f}  hit@3={r['hit@3']:.3f}  MRR={r['mrr']:.3f}")
        for q, exp, got in r["misses"]:
            print(f"  ✗ {q!r}: beklenen {exp}, ilk 3: {got}")

    damaged = [h.doc for h in main_idx.search("hasarlı ürün", k=3)]
    ok = "04_hasarli_yanlis_urun.md" in damaged
    print(f"\n{'✓' if ok else '✗'} search_policies('hasarlı ürün') ilk 3: {damaged}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["embedder", "n", "hit@1", "hit@3", "mrr"])
        for r in rows:
            w.writerow([r["embedder"], r["n"], r["hit@1"], r["hit@3"], r["mrr"]])
    print(f"Sonuçlar: {RESULTS}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
