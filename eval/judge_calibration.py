"""Hakem (LLM-as-judge) kalibrasyonu: kasıtlı yanlış cevapları yakalıyor mu?

Kullanım:
    python eval/judge_calibration.py           # 5 hakem çağrısı (JUDGE_MODEL)

Neden: Golden set'teki politika cevaplarının hepsi doğru çıktığında hakem her şeye 1,0 verir;
bu, hakemin ayırt edici olduğunu göstermez. Burada bağlam aynı kalır, cevaba bilinen bir hata
konur (yanlış süre, yanlış eşik, politikada olmayan iddia). Hakem doğru cevaplara 1,0, yanlışlara
< 1,0 vermeli. [Ref: Genel Rehber §7 "hakemin kararını … elle kontrol et ve uyum oranını yaz"]
Sonuç: eval/results/judge_calibration.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import config  # noqa: E402
from run_eval import JUDGE_PROMPT, build_llm  # noqa: E402
from src.knowledge.policy_search import load_chunks  # noqa: E402
from src.llm_client import LLMError  # noqa: E402

# (kimlik, doküman, bölüm, soru, cevap, cevap politikaya sadık mı?)
CASES = [
    ("C1", "04_hasarli_yanlis_urun.md", "Süre", "Hasarlı ürünü kaç gün içinde iade edebilirim?",
     "Teslimden itibaren 30 gün içinde iade talebi açabilirsiniz; 30. gün dahildir.", True),
    ("C2", "04_hasarli_yanlis_urun.md", "Süre", "Hasarlı ürünü kaç gün içinde iade edebilirim?",
     "Teslimden itibaren 60 gün içinde iade talebi açabilirsiniz.", False),          # yanlış süre
    ("C3", "05_odeme_ve_geri_odeme.md", "Onay seviyeleri", "Büyük iadeler nasıl onaylanır?",
     "500 BRL'yi aşan iadeler kıdemli onay gerektirir.", True),
    ("C4", "05_odeme_ve_geri_odeme.md", "Onay seviyeleri", "Büyük iadeler nasıl onaylanır?",
     "1000 BRL'yi aşan iadeler kıdemli onay gerektirir.", False),                   # yanlış eşik
    ("C5", "02_iptal_politikasi.md", "Kargodaki siparişler", "Kargodaki siparişi iptal edebilir miyim?",
     "Evet, kargodaki siparişi 24 saat içinde ücretsiz iptal edebilirsiniz.", False),  # politikada yok
]


def main() -> int:
    chunks = {(c.doc, c.section): c for c in load_chunks(config.POLICIES_DIR)}
    judge = build_llm(config.JUDGE_MODEL, rpm=config.JUDGE_RPM, rpd=config.JUDGE_RPD)
    rows, agree = [], 0
    for cid, doc, section, question, answer, faithful in CASES:
        chunk = chunks[(doc, section)]
        context = f"[{doc} — {section}]\n{chunk.text}"
        try:
            resp = judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(
                context=context, question=question, answer=answer)}])
            data = json.loads(resp.content[resp.content.index("{"): resp.content.rindex("}") + 1])
            claims, supported = int(data["claims"]), int(data["supported"])
            score = 1.0 if claims == 0 else supported / claims
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            print(f"  ! {cid}: {type(exc).__name__}")
            rows.append({"id": cid, "expected_faithful": faithful, "score": None})
            continue
        ok = (score == 1.0) == faithful
        agree += ok
        rows.append({"id": cid, "expected_faithful": faithful, "score": round(score, 2), "agree": ok,
                     "unsupported": data.get("unsupported_examples", [])[:2]})
        print(f"  {'✓' if ok else '✗'} {cid} beklenen={'sadık' if faithful else 'sadık değil'} "
              f"hakem={score:.2f}")
    judged = sum(r["score"] is not None for r in rows)
    print(f"\nHakem-beklenti uyumu: {agree}/{judged}")
    out = ROOT / "eval" / "results" / "judge_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"judge_model": config.JUDGE_MODEL, "agreement": f"{agree}/{judged}",
                               "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sonuç: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
