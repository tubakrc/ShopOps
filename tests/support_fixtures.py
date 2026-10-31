"""Tool ve agent testleri için küçük, bilinen içerikli SQLite DB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from load_sqlite import RETURN_REQUESTS_DDL, SCHEMA


def hexid(n: int) -> str:
    return f"{n:032x}"


O_ELIG = hexid(1)        # 2018-08-25 teslim, 300 + 40 kargo
O_BIG = hexid(2)         # 600 + 20 → PENDING_SENIOR
O_SHIP = hexid(3)        # shipped
O_FUTURE = hexid(4)      # delivered ama 2018-09-05 → yolda
O_FREIGHT = hexid(5)     # 22. gün, 16 gün geç → yalnız kargo
FOREIGN = [hexid(100 + i) for i in range(10)]
NONEXISTENT = hexid(999)
AYSE = "u-ayse"

ORDERS = [
    # order_id, customer_id, uid, status, purchase, carrier, delivered, estimated, price, freight
    (O_ELIG, "c1", AYSE, "delivered", "2018-08-15 10:00:00", "2018-08-16 10:00:00",
     "2018-08-25 15:00:00", "2018-08-30 00:00:00", 300.0, 40.0),
    (O_BIG, "c2", AYSE, "delivered", "2018-08-20 10:00:00", "2018-08-21 10:00:00",
     "2018-08-28 15:00:00", "2018-09-03 00:00:00", 600.0, 20.0),
    (O_SHIP, "c3", AYSE, "shipped", "2018-08-29 10:00:00", "2018-08-30 10:00:00",
     None, "2018-09-12 00:00:00", 80.0, 10.0),
    (O_FUTURE, "c4", AYSE, "delivered", "2018-08-27 10:00:00", "2018-08-28 10:00:00",
     "2018-09-05 12:00:00", "2018-09-10 00:00:00", 50.0, 8.0),
    (O_FREIGHT, "c5", AYSE, "delivered", "2018-07-01 10:00:00", "2018-07-02 10:00:00",
     "2018-08-10 12:00:00", "2018-07-25 00:00:00", 120.0, 25.5),
] + [(oid, f"cx{i}", "u-other", "delivered", "2018-08-15 10:00:00", "2018-08-16 10:00:00",
      "2018-08-25 15:00:00", "2018-08-30 00:00:00", 100.0, 10.0) for i, oid in enumerate(FOREIGN)]


def build_support_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    for name, (_, cols, pk) in SCHEMA.items():
        pk_sql = f", PRIMARY KEY ({pk})" if pk else ""
        con.execute(f"CREATE TABLE {name} ({', '.join(f'{n} {t}' for n, t in cols)}{pk_sql})")
    con.executescript(RETURN_REQUESTS_DDL)
    con.execute("INSERT INTO products (product_id, product_category_name) VALUES ('p1', 'moveis_decoracao')")
    con.execute("INSERT INTO category_translation VALUES ('moveis_decoracao', 'furniture_decor')")
    for oid, cid, uid, st, pur, car, dlv, est, price, freight in ORDERS:
        con.execute("INSERT INTO customers VALUES (?,?,?,?,?)", (cid, uid, "01000", "sao paulo", "SP"))
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", (oid, cid, st, pur, pur, car, dlv, est))
        con.execute("INSERT INTO order_items VALUES (?,?,?,?,?,?,?)",
                    (oid, 1, "p1", "s1", "2018-12-31 00:00:00", price, freight))
        con.execute("INSERT INTO order_payments VALUES (?,?,?,?,?)", (oid, 1, "credit_card", 1, price + freight))
    con.commit()
    con.close()
    return path
