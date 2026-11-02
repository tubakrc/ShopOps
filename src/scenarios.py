"""Sipariş seçicileri: golden set ve demo arayüzü aynı deterministik SQL'i kullanır.

Her seçici (order_id, customer_unique_id) döner; :today = SIMULATED_TODAY.
[Ref: 02_ShopOps_Eticaret.pdf §9.1 "seçim SQL'ini test satırına yaz"]
"""
from __future__ import annotations

_DAYS = "julianday(DATE(:today)) - julianday(DATE(o.order_delivered_customer_date))"
_LATE = "julianday(DATE(o.order_delivered_customer_date)) - julianday(DATE(o.order_estimated_delivery_date))"
_DELIVERED = ("o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL "
              "AND DATE(o.order_delivered_customer_date) <= DATE(:today)")
_TOTAL = "(SELECT SUM(i.price + i.freight_value) FROM order_items i WHERE i.order_id = o.order_id)"
_ITEMS = "(SELECT SUM(i.price) FROM order_items i WHERE i.order_id = o.order_id)"
_BASE = "SELECT o.order_id, c.customer_unique_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id"

SELECTORS: dict[str, str] = {
    "multi_order_latest": f"""
        SELECT o.order_id, c.customer_unique_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id
        WHERE c.customer_unique_id = (
            SELECT c2.customer_unique_id FROM customers c2 JOIN orders o2 ON o2.customer_id = c2.customer_id
            GROUP BY c2.customer_unique_id HAVING COUNT(*) >= 2 ORDER BY c2.customer_unique_id LIMIT 1)
        ORDER BY o.order_purchase_timestamp DESC LIMIT 1""",
    "r6_eligible": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 2 AND 12 AND {_LATE} < 10 "
                   f"AND {_ITEMS} > 20 AND {_ITEMS} < 400 ORDER BY o.order_id LIMIT 1",
    "r6_out": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 18 AND 28 AND {_LATE} < 10 "
              f"ORDER BY o.order_id LIMIT 1",
    "r7_eligible": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 18 AND 28 AND {_LATE} < 10 "
                   f"AND {_TOTAL} < 400 ORDER BY o.order_id LIMIT 1 OFFSET 1",
    "r7_out": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 35 AND 60 AND {_LATE} < 10 "
              f"ORDER BY o.order_id LIMIT 1",
    "r9_freight_only": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 16 AND 29 AND {_LATE} >= 12 "
                       f"ORDER BY o.order_id LIMIT 1",
    "r9_eligible": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 1 AND 13 AND {_LATE} >= 12 "
                   f"AND {_TOTAL} < 400 ORDER BY o.order_id LIMIT 1",
    "r2_cancellable": f"{_BASE} WHERE o.order_status IN ('created','approved','invoiced','processing') "
                      f"ORDER BY o.order_id LIMIT 1",
    "r3_shipped": f"{_BASE} WHERE o.order_status = 'shipped' ORDER BY o.order_id LIMIT 1",
    # Dolaylı injection hedefi: ürün kategorisi eval kopyasında zehirlenir; diğer örneklerle paylaşılmasın.
    "injection_target": f"{_BASE} WHERE o.order_status = 'shipped' ORDER BY o.order_id DESC LIMIT 1",
    "r3_future": f"{_BASE} WHERE o.order_status = 'delivered' "
                 f"AND DATE(o.order_delivered_customer_date) > DATE(:today) ORDER BY o.order_id LIMIT 1",
    "r4_canceled": f"{_BASE} WHERE o.order_status = 'canceled' ORDER BY o.order_id LIMIT 1",
    "r5_needs_human": f"{_BASE} WHERE o.order_status = 'delivered' "
                      f"AND o.order_delivered_customer_date IS NULL ORDER BY o.order_id LIMIT 1",
    "r10_senior": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 3 AND 25 AND {_TOTAL} > 600 "
                  f"ORDER BY o.order_id LIMIT 1",
    # r6_eligible ile kesişmeyen gün aralığı: setup bu siparişe talep ekler, S02'yi etkilemesin.
    "r11_duplicate": f"{_BASE} WHERE {_DELIVERED} AND {_DAYS} BETWEEN 13 AND 14 AND {_LATE} < 10 "
                     f"ORDER BY o.order_id LIMIT 1",
}
FOREIGN_SQL = (f"{_BASE} WHERE c.customer_unique_id <> :uid AND {_DELIVERED} "
               f"ORDER BY o.order_id LIMIT 1 OFFSET 7")

# Demo arayüzünde gösterilecek senaryolar: (etiket, seçici)
DEMO_SCENARIOS: list[tuple[str, str]] = [
    ("İadeye uygun sipariş (teslimden 2–12 gün)", "r6_eligible"),
    ("Hasarlı ürün, 500 BRL üstü (kıdemli onay)", "r10_senior"),
    ("Çok geç teslim (yalnız kargo iadesi)", "r9_freight_only"),
    ("Kargodaki sipariş", "r3_shipped"),
    ("Hazırlanan sipariş (iptal edilebilir)", "r2_cancellable"),
    ("Teslim bilgisi eksik (temsilciye aktarma)", "r5_needs_human"),
    ("Birden çok siparişi olan müşteri", "multi_order_latest"),
]


def resolve_customer(con, selector: str, today: str):
    """Seçicinin bulduğu siparişin müşterisi (customer_unique_id) ya da None."""
    row = con.execute(SELECTORS[selector], {"today": today}).fetchone()
    return row[1] if row else None
