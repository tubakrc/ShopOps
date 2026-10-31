"""Olist CSV'lerini SQLite'a yükler, indeksler ve boş return_requests tablosunu kurar.

Kullanım:
    python scripts/load_sqlite.py --src <kagglehub_yolu> --db data/shopops.db [--force] [--with-geolocation]

Kurallar:
- R0: order_purchase_timestamp >= SIMULATED_TODAY olan siparişler (ve onlara bağlı
  kalem/ödeme/yorum/müşteri satırları) yüklenmez. [Ref: ShopOps §6.1 R0]
- Boş string → NULL. Tarihler ISO metin olarak kalır; karşılaştırma DATE()/julianday()
  ile yapılır. [Ref: ShopOps §11 "Tarihler metin"]
- Ham tablolara sonradan yazılmaz; tek yazma hedefi return_requests. [Ref: ShopOps §3]
- Geolocation (1M satır) Temel seviyede gerekmez; isteğe bağlı. [Ref: ShopOps §3]
- DB önce geçici dosyaya kurulur, başarılıysa yerine taşınır (yarım DB kalmaz).
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# tablo -> (csv dosyası, [(kolon, tip)], birincil anahtar)
SCHEMA: dict[str, tuple[str, list[tuple[str, str]], Optional[str]]] = {
    "orders": ("olist_orders_dataset.csv", [
        ("order_id", "TEXT"), ("customer_id", "TEXT"), ("order_status", "TEXT"),
        ("order_purchase_timestamp", "TEXT"), ("order_approved_at", "TEXT"),
        ("order_delivered_carrier_date", "TEXT"), ("order_delivered_customer_date", "TEXT"),
        ("order_estimated_delivery_date", "TEXT"),
    ], "order_id"),
    "order_items": ("olist_order_items_dataset.csv", [
        ("order_id", "TEXT"), ("order_item_id", "INTEGER"), ("product_id", "TEXT"),
        ("seller_id", "TEXT"), ("shipping_limit_date", "TEXT"), ("price", "REAL"),
        ("freight_value", "REAL"),
    ], "order_id, order_item_id"),
    "order_payments": ("olist_order_payments_dataset.csv", [
        ("order_id", "TEXT"), ("payment_sequential", "INTEGER"), ("payment_type", "TEXT"),
        ("payment_installments", "INTEGER"), ("payment_value", "REAL"),
    ], "order_id, payment_sequential"),
    # review_id Olist'te tekil değil; birincil anahtar yok.
    "order_reviews": ("olist_order_reviews_dataset.csv", [
        ("review_id", "TEXT"), ("order_id", "TEXT"), ("review_score", "INTEGER"),
        ("review_comment_title", "TEXT"), ("review_comment_message", "TEXT"),
        ("review_creation_date", "TEXT"), ("review_answer_timestamp", "TEXT"),
    ], None),
    "customers": ("olist_customers_dataset.csv", [
        ("customer_id", "TEXT"), ("customer_unique_id", "TEXT"),
        ("customer_zip_code_prefix", "TEXT"), ("customer_city", "TEXT"), ("customer_state", "TEXT"),
    ], "customer_id"),
    "sellers": ("olist_sellers_dataset.csv", [
        ("seller_id", "TEXT"), ("seller_zip_code_prefix", "TEXT"), ("seller_city", "TEXT"),
        ("seller_state", "TEXT"),
    ], "seller_id"),
    "products": ("olist_products_dataset.csv", [
        ("product_id", "TEXT"), ("product_category_name", "TEXT"),
        ("product_name_lenght", "INTEGER"), ("product_description_lenght", "INTEGER"),
        ("product_photos_qty", "INTEGER"), ("product_weight_g", "REAL"),
        ("product_length_cm", "REAL"), ("product_height_cm", "REAL"), ("product_width_cm", "REAL"),
    ], "product_id"),
    "category_translation": ("product_category_name_translation.csv", [
        ("product_category_name", "TEXT"), ("product_category_name_english", "TEXT"),
    ], "product_category_name"),
}

GEOLOCATION = ("olist_geolocation_dataset.csv", [
    ("geolocation_zip_code_prefix", "TEXT"), ("geolocation_lat", "REAL"),
    ("geolocation_lng", "REAL"), ("geolocation_city", "TEXT"), ("geolocation_state", "TEXT"),
], None)

# Sipariş başına bağlı tablolar: R0'da dışlanan siparişlerin satırları da düşer.
ORDER_SCOPED = {"order_items", "order_payments", "order_reviews"}

INDEXES = [
    "CREATE INDEX ix_orders_customer ON orders(customer_id)",
    "CREATE INDEX ix_orders_status ON orders(order_status)",
    "CREATE INDEX ix_orders_purchase ON orders(order_purchase_timestamp)",
    "CREATE INDEX ix_customers_unique ON customers(customer_unique_id)",
    "CREATE INDEX ix_items_order ON order_items(order_id)",
    "CREATE INDEX ix_items_seller ON order_items(seller_id)",
    "CREATE INDEX ix_items_product ON order_items(product_id)",
    "CREATE INDEX ix_payments_order ON order_payments(order_id)",
    "CREATE INDEX ix_reviews_order ON order_reviews(order_id)",
]

# Tek yazma hedefi. Statüler [Ref: ShopOps §6.1 R10, R11]; HITL onaylayan approved_by'da.
RETURN_REQUESTS_DDL = """
CREATE TABLE return_requests (
    return_request_id   TEXT PRIMARY KEY,
    order_id            TEXT NOT NULL REFERENCES orders(order_id),
    customer_unique_id  TEXT NOT NULL,
    reason              TEXT NOT NULL CHECK (reason IN ('changed_mind','damaged','wrong_item','late')),
    refund_amount       REAL NOT NULL CHECK (refund_amount >= 0),
    freight_only        INTEGER NOT NULL DEFAULT 0 CHECK (freight_only IN (0, 1)),
    status              TEXT NOT NULL CHECK (status IN
                          ('PENDING_REVIEW','PENDING_SENIOR','APPROVED','REJECTED')),
    created_on          TEXT NOT NULL,
    approved_by         TEXT NOT NULL
);
CREATE INDEX ix_returns_order ON return_requests(order_id);
"""


def _convert(value: str, sql_type: str) -> object:
    """CSV hücresini SQL tipine çevirir; boş string NULL olur."""
    if value is None or value.strip() == "":
        return None
    if sql_type == "INTEGER":
        return int(float(value))
    if sql_type == "REAL":
        return float(value)
    return value


def _read_rows(path: Path, columns: list[tuple[str, str]]) -> Iterator[tuple]:
    # utf-8-sig: çeviri dosyasının başında BOM var.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c, _ in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: beklenen kolon(lar) yok: {missing}")
        for row in reader:
            yield tuple(_convert(row[c], t) for c, t in columns)


def _create_table(con: sqlite3.Connection, name: str, columns: list[tuple[str, str]],
                  pk: Optional[str]) -> None:
    cols = ", ".join(f"{c} {t}" for c, t in columns)
    pk_sql = f", PRIMARY KEY ({pk})" if pk else ""
    con.execute(f"CREATE TABLE {name} ({cols}{pk_sql})")


def excluded_order_ids(src: Path, today_iso: str) -> set[str]:
    """R0: SIMULATED_TODAY ve sonrasında verilmiş siparişler."""
    fname, columns, _ = SCHEMA["orders"]
    idx_id = [c for c, _ in columns].index("order_id")
    idx_ts = [c for c, _ in columns].index("order_purchase_timestamp")
    return {r[idx_id] for r in _read_rows(src / fname, columns)
            if r[idx_ts] is not None and str(r[idx_ts])[:10] >= today_iso}


def build_database(src: Path, db_path: Path, with_geolocation: bool = False) -> dict[str, int]:
    """CSV'lerden DB'yi kurar. Dönüş: tablo -> yüklenen satır sayısı (+ 'r0_excluded')."""
    today_iso = config.SIMULATED_TODAY.isoformat()
    excluded = excluded_order_ids(src, today_iso)

    # R0'da dışlanan siparişlerin customer_id'leri (Olist'te customer_id sipariş başınadır).
    o_cols = [c for c, _ in SCHEMA["orders"][1]]
    excluded_customers = {r[o_cols.index("customer_id")]
                          for r in _read_rows(src / SCHEMA["orders"][0], SCHEMA["orders"][1])
                          if r[o_cols.index("order_id")] in excluded}

    tmp = db_path.with_suffix(db_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)

    tables = dict(SCHEMA)
    if with_geolocation:
        tables["geolocation"] = GEOLOCATION

    counts: dict[str, int] = {"r0_excluded": len(excluded)}
    con = sqlite3.connect(tmp)
    try:
        for name, (fname, columns, pk) in tables.items():
            _create_table(con, name, columns, pk)
            names = [c for c, _ in columns]
            rows = _read_rows(src / fname, columns)
            if name == "orders" or name in ORDER_SCOPED:
                i = names.index("order_id")
                rows = (r for r in rows if r[i] not in excluded)
            elif name == "customers":
                i = names.index("customer_id")
                rows = (r for r in rows if r[i] not in excluded_customers)
            placeholders = ", ".join("?" for _ in columns)
            cur = con.executemany(f"INSERT INTO {name} VALUES ({placeholders})", rows)
            counts[name] = cur.rowcount
        for ddl in INDEXES:
            con.execute(ddl)
        con.executescript(RETURN_REQUESTS_DDL)
        con.commit()
    except Exception:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    tmp.replace(db_path)
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, type=Path, help="Olist CSV klasörü")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument("--force", action="store_true",
                        help="Var olan DB'nin üzerine yaz (return_requests satırları silinir)")
    parser.add_argument("--with-geolocation", action="store_true")
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"HATA: {args.src} bir klasör değil.", file=sys.stderr)
        return 2
    if args.db.exists() and not args.force:
        print(f"HATA: {args.db} zaten var. Yeniden kurmak için --force ekle.", file=sys.stderr)
        return 2

    counts = build_database(args.src, args.db, args.with_geolocation)
    for name, n in counts.items():
        print(f"{name:<22} {n:>10,}")
    print(f"\nDB hazır: {args.db}  (SIMULATED_TODAY={config.SIMULATED_TODAY})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
