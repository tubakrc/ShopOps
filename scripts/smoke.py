"""Yüklenen DB'yi ShopOps belgesindeki sayılarla karşılaştırır.

Kullanım:
    python scripts/smoke.py --db data/shopops.db [--src <csv_klasörü>]

--src verilirse önce ham CSV satır sayıları da kontrol edilir. [Ref: ShopOps §3]
DB kontrolleri, kural ve metrik tanımlarının veriyle tutarlı olduğunu kanıtlar.
[Ref: ShopOps §5.4, §6.1, §9.1]  Herhangi bir kontrol başarısızsa çıkış kodu 1'dir.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# [Ref: ShopOps §3 tablo]
RAW_CSV_ROWS: dict[str, int] = {
    "olist_orders_dataset.csv": 99_441,
    "olist_order_items_dataset.csv": 112_650,
    "olist_order_payments_dataset.csv": 103_886,
    "olist_order_reviews_dataset.csv": 99_224,
    "olist_customers_dataset.csv": 99_441,
    "olist_sellers_dataset.csv": 3_095,
    "olist_products_dataset.csv": 32_951,
    "product_category_name_translation.csv": 71,
    "olist_geolocation_dataset.csv": 1_000_163,
}

DELIVERED_PAST = ("o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL "
                  "AND DATE(o.order_delivered_customer_date) <= DATE(:today)")
DAYS_SINCE_DELIVERY = "julianday(DATE(:today)) - julianday(DATE(o.order_delivered_customer_date))"


@dataclass(frozen=True)
class Check:
    name: str
    sql: str
    expected: float
    ref: str
    ndigits: Optional[int] = None  # None: tam sayı eşitliği


CHECKS: list[Check] = [
    Check("orders (R0 sonrası)", "SELECT COUNT(*) FROM orders o", 99_421, "§3 99.441 − R0 20"),
    Check("R0: bugün/sonrası sipariş yok",
          "SELECT COUNT(*) FROM orders o WHERE DATE(o.order_purchase_timestamp) >= DATE(:today)",
          0, "§6.1 R0"),
    Check("customers (R0 sonrası)", "SELECT COUNT(*) FROM customers", 99_421, "customer_id sipariş başına"),
    Check("return_requests boş", "SELECT COUNT(*) FROM return_requests", 0, "§3 tek yazma hedefi"),
    Check("R2: iptal edilebilir durumlar",
          "SELECT COUNT(*) FROM orders o WHERE o.order_status IN "
          "('created','approved','invoiced','processing')", 622, "§9.1"),
    Check("R3: shipped", "SELECT COUNT(*) FROM orders o WHERE o.order_status = 'shipped'",
          1_106, "§9.1"),
    Check("R3: teslim tarihi gelecekte",
          "SELECT COUNT(*) FROM orders o WHERE o.order_status = 'delivered' "
          "AND DATE(o.order_delivered_customer_date) > DATE(:today)", 53, "§6.1 R3"),
    Check("R5: delivered ama tarih boş",
          "SELECT COUNT(*) FROM orders o WHERE o.order_status = 'delivered' "
          "AND o.order_delivered_customer_date IS NULL", 8, "§6.1 R5"),
    Check("late_delivery_rate n", f"SELECT COUNT(*) FROM orders o WHERE {DELIVERED_PAST}",
          96_417, "§5.4"),
    Check("late_delivery_rate",
          "SELECT AVG(CASE WHEN DATE(o.order_delivered_customer_date) > "
          f"DATE(o.order_estimated_delivery_date) THEN 1.0 ELSE 0 END) FROM orders o WHERE {DELIVERED_PAST}",
          0.0672, "§5.4", ndigits=4),
    Check("avg_delivery_days",
          "SELECT AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)) "
          f"FROM orders o WHERE {DELIVERED_PAST}", 12.540, "§5.4", ndigits=3),
    Check("R6 penceresi (0–14 gün)",
          f"SELECT COUNT(*) FROM orders o WHERE {DELIVERED_PAST} AND {DAYS_SINCE_DELIVERY} BETWEEN 0 AND 14",
          3_656, "§9.1"),
    Check("R7 penceresi (0–30 gün)",
          f"SELECT COUNT(*) FROM orders o WHERE {DELIVERED_PAST} AND {DAYS_SINCE_DELIVERY} BETWEEN 0 AND 30",
          8_062, "§9.1"),
    Check("R9: ≥10 gün geç teslim",
          f"SELECT COUNT(*) FROM orders o WHERE {DELIVERED_PAST} AND "
          "julianday(DATE(o.order_delivered_customer_date)) - "
          "julianday(DATE(o.order_estimated_delivery_date)) >= 10", 2_256, "§9.1"),
    Check("R12: voucher kullanılan sipariş",
          "SELECT COUNT(DISTINCT p.order_id) FROM order_payments p WHERE p.payment_type = 'voucher'",
          3_847, "§9.1"),
]


def run_checks(con: sqlite3.Connection, today_iso: str) -> list[tuple[Check, float, bool]]:
    """Her kontrolü çalıştırır: (kontrol, gerçek değer, geçti mi)."""
    results = []
    for chk in CHECKS:
        params = {"today": today_iso} if ":today" in chk.sql else {}
        actual = con.execute(chk.sql, params).fetchone()[0] or 0
        if chk.ndigits is None:
            ok = int(actual) == int(chk.expected)
        else:
            ok = round(float(actual), chk.ndigits) == round(chk.expected, chk.ndigits)
        results.append((chk, actual, ok))
    return results


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ShopOps veri smoke testi")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--src", type=Path, default=None, help="Ham CSV klasörü (isteğe bağlı)")
    args = parser.parse_args(argv)

    failures = 0
    if args.src:
        print("Ham CSV satır sayıları [§3]")
        for fname, expected in RAW_CSV_ROWS.items():
            path = args.src / fname
            if not path.exists():
                print(f"  ✗ {fname}: dosya yok")
                failures += 1
                continue
            actual = count_csv_rows(path)
            ok = actual == expected
            failures += not ok
            print(f"  {'✓' if ok else '✗'} {fname:<42} {actual:>10,}  (beklenen {expected:,})")
        print()

    if not args.db.exists():
        print(f"HATA: {args.db} yok. Önce scripts/load_sqlite.py çalıştır.", file=sys.stderr)
        return 2

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    print(f"DB kontrolleri (SIMULATED_TODAY={config.SIMULATED_TODAY})")
    for chk, actual, ok in run_checks(con, config.SIMULATED_TODAY.isoformat()):
        failures += not ok
        shown = f"{actual:,.4f}" if chk.ndigits else f"{int(actual):,}"
        print(f"  {'✓' if ok else '✗'} {chk.name:<34} {shown:>12}  (beklenen {chk.expected:,}; {chk.ref})")
    con.close()

    print(f"\n{'TÜMÜ GEÇTİ' if failures == 0 else f'{failures} KONTROL BAŞARISIZ'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
