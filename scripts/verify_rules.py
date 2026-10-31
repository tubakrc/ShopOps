"""Kural motorunu yüklenmiş DB'nin tamamında çalıştırır ve belgedeki kontrol değerleriyle karşılaştırır.

Kullanım:
    python scripts/verify_rules.py --db data/shopops.db

Birim testleri kuralların mantığını, bu script ise gerçek veriyle tutarlılığını kanıtlar.
[Ref: 02_ShopOps_Eticaret.pdf §9.1 "Kontrol değeri: … 130 … 352"; R9 dağılımı 33 / 28 / 2.195]
Burada R1 (yetki) devre dışıdır: gerçekler müşteri filtresi olmadan toplu okunur.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.returns import Decision, Item, OrderFacts, evaluate, to_day  # noqa: E402


def load_all_facts(con: sqlite3.Connection) -> list[OrderFacts]:
    """Tüm siparişlerin gerçeklerini 4 toplu sorguyla okur (sipariş başına sorgu yok)."""
    items: dict[str, list[Item]] = defaultdict(list)
    for oid, price, freight, limit in con.execute(
            "SELECT order_id, price, freight_value, shipping_limit_date FROM order_items"):
        items[oid].append(Item(float(price or 0), float(freight or 0), to_day(limit)))
    pays: dict[str, tuple[float, float]] = {
        oid: (float(t), float(v)) for oid, t, v in con.execute(
            "SELECT order_id, SUM(payment_value), "
            "COALESCE(SUM(CASE WHEN payment_type = 'voucher' THEN payment_value END), 0) "
            "FROM order_payments GROUP BY order_id")}
    reqs: dict[str, list[str]] = defaultdict(list)
    for oid, status in con.execute("SELECT order_id, status FROM return_requests"):
        reqs[oid].append(status)

    out = []
    for oid, st, pur, dlv, est, car in con.execute(
            "SELECT order_id, order_status, order_purchase_timestamp, "
            "order_delivered_customer_date, order_estimated_delivery_date, "
            "order_delivered_carrier_date FROM orders"):
        total, voucher = pays.get(oid, (0.0, 0.0))
        out.append(OrderFacts(oid, st, to_day(pur), to_day(dlv), to_day(est), to_day(car),
                              tuple(items[oid]), total, voucher, tuple(reqs[oid])))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Kural motoru veri doğrulaması")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f"HATA: {args.db} yok. Önce scripts/load_sqlite.py çalıştır.", file=sys.stderr)
        return 2

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    all_facts = load_all_facts(con)
    con.close()
    today = config.SIMULATED_TODAY

    cm = [evaluate(f, "changed_mind", today) for f in all_facts]
    dm = [evaluate(f, "damaged", today) for f in all_facts]
    cm_count = Counter(r.decision for r in cm)
    dm_count = Counter(r.decision for r in dm)

    r9 = [r for r in cm if r.freight_refund_flag]
    r9_le14 = sum(r.days_since_delivery <= 14 for r in r9)
    r9_15_30 = sum(15 <= r.days_since_delivery <= 30 for r in r9)
    r9_gt30 = sum(r.days_since_delivery > 30 for r in r9)

    checks = [
        ("R0: kalan NOT_FOUND", cm_count[Decision.NOT_FOUND], 0, "§6.1 R0 (yükleyici dışladı)"),
        ("R2: NOT_DELIVERED_CANCELLABLE", cm_count[Decision.NOT_DELIVERED_CANCELLABLE], 622, "§9.1"),
        ("R3: IN_TRANSIT", cm_count[Decision.IN_TRANSIT], 1_159, "§9.1 1.106 + 53"),
        ("R5: NEEDS_HUMAN", cm_count[Decision.NEEDS_HUMAN], 8, "§9.1"),
        ("R6: changed_mind ELIGIBLE", cm_count[Decision.ELIGIBLE], 3_656, "§9.1"),
        ("R7: damaged ELIGIBLE", dm_count[Decision.ELIGIBLE], 8_062, "§9.1"),
        ("R8/R10: changed_mind tutar > 500",
         sum(r.decision is Decision.ELIGIBLE and r.refund_amount > 500 for r in cm), 130, "§9.1"),
        ("R8/R10: damaged tutar > 500",
         sum(r.decision is Decision.ELIGIBLE and r.refund_amount > 500 for r in dm), 352, "§9.1"),
        ("R9: bayraklı toplam", len(r9), 2_256, "§9.1"),
        ("R9: 0–14 gün (tam iade + kargo)", r9_le14, 33, "§9.1"),
        ("R9: 15–30 gün (yalnız kargo)", r9_15_30, 28, "§9.1"),
        ("R9: >30 gün (talep yok)", r9_gt30, 2_195, "§9.1"),
        ("R9: yalnız-kargo talebi açılabilen",
         sum(r.freight_only and r.can_open_request for r in cm), 28, "§9.1 (changed_mind)"),
    ]

    failures = 0
    print(f"Kural motoru doğrulaması ({len(all_facts):,} sipariş, SIMULATED_TODAY={today})")
    for name, actual, expected, ref in checks:
        ok = actual == expected
        failures += not ok
        print(f"  {'✓' if ok else '✗'} {name:<36} {actual:>8,}  (beklenen {expected:,}; {ref})")

    print("\nBilgi (belgede kontrol değeri yok):")
    print(f"  R4 ALREADY_CLOSED                     {cm_count[Decision.ALREADY_CLOSED]:>8,}")
    print(f"  R13 satıcı kaynaklı gecikme bayrağı   {sum(r.seller_caused_delay for r in cm):>8,}"
          "  (teslim edilmiş, ≤ bugün)")
    senior = sum(r.approval_status is not None and r.approval_status.value == "PENDING_SENIOR" for r in dm)
    print(f"  R10 damaged PENDING_SENIOR            {senior:>8,}")

    print(f"\n{'TÜMÜ GEÇTİ' if failures == 0 else f'{failures} KONTROL BAŞARISIZ'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
