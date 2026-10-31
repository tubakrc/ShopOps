"""load_sqlite.py testleri: R0 dışlama, NULL dönüşümü, return_requests, atomik kurulum."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from load_sqlite import build_database, main


@pytest.fixture
def db(olist_dir: Path, tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "shopops.db"
    build_database(olist_dir, path)
    con = sqlite3.connect(path)
    yield con
    con.close()


def test_r0_excludes_orders_on_or_after_simulated_today(db):
    ids = {r[0] for r in db.execute("SELECT order_id FROM orders")}
    assert ids == {"o1", "o2"}


def test_r0_also_drops_dependent_rows(db):
    for table in ("order_items", "order_payments", "order_reviews"):
        assert db.execute(f"SELECT COUNT(*) FROM {table} WHERE order_id = 'o3'").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM customers WHERE customer_id = 'c3'").fetchone()[0] == 0


def test_empty_string_becomes_null(db):
    delivered = db.execute(
        "SELECT order_delivered_customer_date FROM orders WHERE order_id = 'o2'").fetchone()[0]
    assert delivered is None


def test_numeric_columns_are_typed(db):
    price, freight = db.execute("SELECT price, freight_value FROM order_items").fetchone()
    assert isinstance(price, float) and price == pytest.approx(120.50)
    assert freight == pytest.approx(15.20)


def test_bom_header_is_read(db):
    row = db.execute("SELECT product_category_name_english FROM category_translation").fetchone()
    assert row == ("furniture_decor",)


def test_customer_unique_id_links_multiple_orders(db):
    orders = db.execute(
        "SELECT o.order_id FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.customer_unique_id = 'u1' ORDER BY o.order_id").fetchall()
    assert [r[0] for r in orders] == ["o1", "o2"]


def test_return_requests_exists_empty_with_constraints(db):
    assert db.execute("SELECT COUNT(*) FROM return_requests").fetchone()[0] == 0
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO return_requests VALUES "
                   "('rr1','o1','u1','changed_mind',10,0,'APPROVED_BY_LLM','2018-09-01','agent')")


def test_existing_db_requires_force(olist_dir: Path, tmp_path: Path, capsys):
    path = tmp_path / "x.db"
    assert main(["--src", str(olist_dir), "--db", str(path)]) == 0
    assert main(["--src", str(olist_dir), "--db", str(path)]) == 2
    assert main(["--src", str(olist_dir), "--db", str(path), "--force"]) == 0
    assert not path.with_suffix(".db.tmp").exists()


def test_missing_column_fails_loudly(olist_dir: Path, tmp_path: Path):
    (olist_dir / "olist_sellers_dataset.csv").write_text("seller_id\ns1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="beklenen kolon"):
        build_database(olist_dir, tmp_path / "y.db")
    assert not (tmp_path / "y.db").exists()
    assert not (tmp_path / "y.db.tmp").exists()
