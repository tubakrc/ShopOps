"""Eval kayıtlarını okunur biçimde gösterir (hata analizi için).

Kullanım:
    python eval/show_records.py --ids A04,A06,A08 --systems b0,b1
    python eval/show_records.py --failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RECORDS = Path(__file__).resolve().parent / "results" / "records.jsonl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="")
    p.add_argument("--systems", default="b0,b1,final")
    p.add_argument("--failed", action="store_true", help="Yalnız task_success=False olanlar")
    p.add_argument("--judged", action="store_true", help="Yalnız hakemin hüküm verdiği örnekler (elle kontrol)")
    p.add_argument("--max-chars", type=int, default=600)
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    ids = {x.strip() for x in args.ids.split(",") if x.strip()}
    systems = {x.strip() for x in args.systems.split(",") if x.strip()}
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["system"] not in systems or (ids and r["id"] not in ids):
            continue
        if args.failed and r["scores"].get("task_success") is not False:
            continue
        if args.judged and r["scores"].get("faithfulness") is None:
            continue
        sc = r["scores"]
        # "fabricated" için True kötüdür; diğer skorlarda False kötüdür.
        failed = [k for k, v in sc.items() if (v is True if k == "fabricated" else v is False)]
        print(f"── {r['system']} {r['id']} · karar={r['decision']} · başarısız={failed or '-'}")
        for t in r["tools"]:
            print(f"   tool: {t['name']}({t['args']}) → {t['code']}")
        if sc.get("faithfulness") is not None:
            print(f"   hakem: faithfulness={sc['faithfulness']:.2f} · desteklenmeyen: "
                  f"{sc.get('judge_unsupported') or '-'}")
        answer = r["answer"].replace("\n", " ")
        print(f"   cevap: {answer[:args.max_chars]}{'…' if len(answer) > args.max_chars else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
