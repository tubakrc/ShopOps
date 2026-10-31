"""ShopOps müşteri destek CLI demosu (Temel seviye arayüzü).

Kullanım:
    python cli.py --demo                 # örnek bir müşteri seçer
    python cli.py --customer <customer_unique_id>
    python cli.py --demo --steps         # her turda tool adımlarını da göster

Kimlik doğrulama burada simüle edilir: oturum, komut satırında seçilen customer_unique_id ile
kurulur; sohbette söylenen hiçbir şey oturumu değiştirmez. [Ref: ShopOps §5.1, §5.3]
İade talebi yalnız buradaki "Onaylıyor musunuz? [e/h]" sorusuna "e" cevabıyla yazılır. [§6.2 G1]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import config
from src.agents.support_agent import AgentTurn, SupportAgent
from src.knowledge.policy_search import open_index
from src.llm_client import LLMClient, LLMError
from src.observability import hash_id, safe_args
from src.tools.support import Session, ToolContext, commit_pending, reject_pending, result_code

DEMO_CUSTOMER_SQL = """
SELECT c.customer_unique_id
FROM customers c JOIN orders o ON o.customer_id = c.customer_id
GROUP BY c.customer_unique_id
HAVING COUNT(*) >= 2
   AND SUM(o.order_status = 'delivered'
           AND julianday(DATE(:today)) - julianday(DATE(o.order_delivered_customer_date)) BETWEEN 0 AND 14) >= 1
ORDER BY c.customer_unique_id LIMIT 1
"""

REASON_TR = {"changed_mind": "fikir değişikliği", "damaged": "hasarlı ürün",
             "wrong_item": "yanlış ürün", "late": "geç teslim"}


def pick_demo_customer(db: Path) -> Optional[str]:
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    row = con.execute(DEMO_CUSTOMER_SQL, {"today": config.SIMULATED_TODAY.isoformat()}).fetchone()
    con.close()
    return row[0] if row else None


def print_steps(turn: AgentTurn) -> None:
    for c in turn.tool_calls:
        print(f"   · {c.name}({safe_args(c.arguments)}) → {result_code(c.result)}")
    s = turn.summary
    print(f"   · {s.get('steps', 0)} LLM adımı, {s.get('prompt_tokens', 0)}+"
          f"{s.get('completion_tokens', 0)} token, {s.get('total_latency_s', 0)} sn")


def ask_approval(agent: SupportAgent) -> None:
    """HITL kapısı. Sonuç agent geçmişine yazılır ki model sonraki turda bunu bilsin."""
    ctx = agent.ctx
    p = ctx.pending
    if p is None:
        return
    print("\n┌─ İADE TALEBİ ONAYI ─────────────────────────────")
    print(f"│ Sipariş       : {p.order_id}")
    print(f"│ Gerekçe       : {REASON_TR.get(p.reason, p.reason)}"
          f"{' (yalnız kargo ücreti)' if p.freight_only else ''}")
    print(f"│ İade tutarı   : {p.refund_amount:.2f} BRL")
    if p.voucher_refund:
        print(f"│   hediye çeki : {p.voucher_refund:.2f} BRL")
    print(f"│   ödeme yönt. : {p.original_method_refund:.2f} BRL")
    print(f"│ Onay kuyruğu  : {p.approval_status}")
    print("└─────────────────────────────────────────────────")
    answer = input("Onaylıyor musunuz? [e/h] ").strip().lower()
    if answer in {"e", "evet", "y", "yes"}:
        out = commit_pending(ctx, p.proposal_id, approved_by="customer-cli")
        if "error" in out:
            note = f"İade talebi oluşturulamadı: {out['message']}"
        else:
            note = (f"İade talebiniz oluşturuldu: {out['return_request_id']} "
                    f"({out['status']}, {out['refund_amount_brl']:.2f} BRL).")
    else:
        reject_pending(ctx)
        note = "İade talebi müşterinin isteğiyle oluşturulmadı."
    print(("✓ " if note.startswith("İade talebiniz") else "✗ ") + note)
    agent.history.append({"role": "assistant", "content": note})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ShopOps müşteri destek CLI")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--customer", help="Oturum müşterisi (customer_unique_id)")
    who.add_argument("--demo", action="store_true", help="Örnek bir müşteri seç")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--steps", action="store_true", help="Tool adımlarını göster")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not args.db.exists():
        print(f"HATA: {args.db} yok. Önce scripts/load_sqlite.py çalıştır.", file=sys.stderr)
        return 2
    uid = args.customer or pick_demo_customer(args.db)
    if not uid:
        print("HATA: uygun demo müşterisi bulunamadı.", file=sys.stderr)
        return 2

    try:
        llm = LLMClient.from_config()
    except LLMError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    index = open_index(config.POLICY_INDEX_PATH, config.POLICIES_DIR, config.EMBED_MODEL)
    ctx = ToolContext(Session("customer", uid), args.db, index,
                      escalation_path=config.ESCALATION_PATH)
    agent = SupportAgent(llm, ctx)

    print(f"ShopOps destek asistanı · bugün {config.SIMULATED_TODAY} · müşteri #{hash_id(uid)}")
    print(f"Model: {llm.model} · politika arama: {index.embedder.name}")
    print("Çıkmak için 'q'.\n")
    try:
        while True:
            try:
                msg = input("Siz: ").strip()
            except EOFError:
                break
            if msg.lower() in {"q", "quit", "exit", "çık"}:
                break
            if not msg:
                continue
            turn = agent.run(msg)
            print(f"\nAsistan: {turn.answer}")
            if args.steps:
                print_steps(turn)
            ask_approval(agent)
            print()
    finally:
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
