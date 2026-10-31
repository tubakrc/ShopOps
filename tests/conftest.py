"""Ortak test yardımcıları: repo kökünü import yoluna ekler, küçük sentetik Olist üretir."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from load_sqlite import SCHEMA  # noqa: E402


def _write(path: Path, header: list[str], rows: list[list[str]], bom: bool = False) -> None:
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def olist_dir(tmp_path: Path) -> Path:
    """3 siparişlik sentetik veri: biri teslim edilmiş, biri yolda, biri R0 kapsamında (2018-09-01)."""
    src = tmp_path / "olist"
    src.mkdir()
    cols = {name: [c for c, _ in spec[1]] for name, spec in SCHEMA.items()}

    _write(src / SCHEMA["orders"][0], cols["orders"], [
        ["o1", "c1", "delivered", "2018-08-01 10:00:00", "2018-08-01 11:00:00",
         "2018-08-03 09:00:00", "2018-08-20 15:00:00", "2018-08-15 00:00:00"],
        ["o2", "c2", "shipped", "2018-08-25 10:00:00", "2018-08-25 11:00:00",
         "2018-08-27 09:00:00", "", "2018-09-10 00:00:00"],
        ["o3", "c3", "created", "2018-09-01 08:00:00", "", "", "", "2018-09-20 00:00:00"],
    ])
    _write(src / SCHEMA["order_items"][0], cols["order_items"], [
        ["o1", "1", "p1", "s1", "2018-08-02 00:00:00", "120.50", "15.20"],
        ["o3", "1", "p1", "s1", "2018-09-03 00:00:00", "80.00", "10.00"],
    ])
    _write(src / SCHEMA["order_payments"][0], cols["order_payments"], [
        ["o1", "1", "credit_card", "2", "135.70"],
        ["o3", "1", "voucher", "1", "90.00"],
    ])
    _write(src / SCHEMA["order_reviews"][0], cols["order_reviews"], [
        ["r1", "o1", "4", "", "chegou certo", "2018-08-21 00:00:00", "2018-08-22 00:00:00"],
        ["r1", "o3", "1", "", "", "2018-09-02 00:00:00", "2018-09-03 00:00:00"],  # tekrar eden review_id
    ])
    _write(src / SCHEMA["customers"][0], cols["customers"], [
        ["c1", "u1", "01000", "sao paulo", "SP"],
        ["c2", "u1", "01000", "sao paulo", "SP"],
        ["c3", "u2", "20000", "rio de janeiro", "RJ"],
    ])
    _write(src / SCHEMA["sellers"][0], cols["sellers"], [["s1", "13000", "campinas", "SP"]])
    _write(src / SCHEMA["products"][0], cols["products"], [
        ["p1", "moveis_decoracao", "40", "300", "2", "1500", "30", "20", "10"],
    ])
    _write(src / SCHEMA["category_translation"][0], cols["category_translation"],
           [["moveis_decoracao", "furniture_decor"]], bom=True)
    return src
