"""Kural motoru testleri: R0–R13'ün her biri için ≥1 test + §6.1 sınır testleri.

[Ref: 02_ShopOps_Eticaret.pdf §6.1, §7 Temel "R0–R13 unit testleri; yetkisiz erişim testi %100"]
"""
from __future__ import annotations

import sqlite3
from datetime import date
from typing import Optional

import pytest

from load_sqlite import RETURN_REQUESTS_DDL, SCHEMA
from src.returns import (ApprovalStatus, Decision, Item, OrderFacts,
                         check_return_eligibility, evaluate, load_order_facts, to_day)

TODAY = date(2018, 9, 1)


def facts(status: str = "delivered", delivered: Optional[str] = "2018-08-25",
          estimated: Optional[str] = "2018-08-30", purchase: str = "2018-08-10",
          carrier: Optional[str] = None, items=((100.0, 20.0, None),),
          payment_total: Optional[float] = None, voucher_total: float = 0.0,
          requests=()) -> OrderFacts:
    its = tuple(Item(p, f, to_day(lim)) for p, f, lim in items)
    pay = sum(p + f for p, f, _ in items) if payment_total is None else payment_total
    return OrderFacts("o1", status, to_day(purchase), to_day(delivered), to_day(estimated),
                      to_day(carrier), its, pay, voucher_total, tuple(requests))


# --------------------------------------------------------------------------- #
# R0, R1
# --------------------------------------------------------------------------- #
def test_r0_order_on_or_after_today_is_not_found():
    r = evaluate(facts(purchase="2018-09-01", delivered=None, status="created"), "changed_mind", TODAY)
    assert r.decision is Decision.NOT_FOUND and r.rule == "R0"


def test_r1_missing_facts_is_not_found_with_no_amount():
    r = evaluate(None, "damaged", TODAY)
    assert r.decision is Decision.NOT_FOUND and r.refund_amount == 0 and r.approval_status is None


# --------------------------------------------------------------------------- #
# R2–R5
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", ["created", "approved", "invoiced", "processing"])
def test_r2_cancellable(status):
    r = evaluate(facts(status=status, delivered=None), "changed_mind", TODAY)
    assert r.decision is Decision.NOT_DELIVERED_CANCELLABLE and r.rule == "R2"


def test_r3_shipped_is_in_transit():
    r = evaluate(facts(status="shipped", delivered=None), "damaged", TODAY)
    assert r.decision is Decision.IN_TRANSIT and not r.can_open_request


def test_r3_boundary_delivered_in_future_is_in_transit():
    # §6.1 sınır testi: teslim 2018-09-05 olan delivered sipariş → IN_TRANSIT
    r = evaluate(facts(delivered="2018-09-05 10:00:00"), "changed_mind", TODAY)
    assert r.decision is Decision.IN_TRANSIT and r.rule == "R3"
    assert r.late_days is None  # gelecekteki teslimden bayrak üretilmez


@pytest.mark.parametrize("status", ["canceled", "unavailable"])
def test_r4_already_closed(status):
    r = evaluate(facts(status=status, delivered=None), "changed_mind", TODAY)
    assert r.decision is Decision.ALREADY_CLOSED and r.rule == "R4"


def test_r5_delivered_without_date_needs_human():
    r = evaluate(facts(delivered=None), "damaged", TODAY)
    assert r.decision is Decision.NEEDS_HUMAN and r.rule == "R5"


def test_unknown_status_goes_to_human_not_guessed():
    assert evaluate(facts(status="lost_in_space"), "damaged", TODAY).decision is Decision.NEEDS_HUMAN


# --------------------------------------------------------------------------- #
# R6 sınırları (changed_mind, late)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delivered,days,expected", [
    ("2018-09-01", 0, Decision.ELIGIBLE),             # teslim günü 0. gün
    ("2018-08-18", 14, Decision.ELIGIBLE),
    ("2018-08-18 23:59:59", 14, Decision.ELIGIBLE),   # saat atılır
    ("2018-08-17", 15, Decision.NOT_ELIGIBLE_WINDOW),
])
@pytest.mark.parametrize("reason", ["changed_mind", "late"])
def test_r6_window_boundaries(delivered, days, expected, reason):
    r = evaluate(facts(delivered=delivered, estimated="2018-09-30"), reason, TODAY)
    assert r.decision is expected and r.rule == "R6" and r.days_since_delivery == days


# --------------------------------------------------------------------------- #
# R7 sınırları (damaged, wrong_item)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delivered,expected", [
    ("2018-08-02", Decision.ELIGIBLE),             # 30. gün
    ("2018-08-01", Decision.NOT_ELIGIBLE_WINDOW),  # 31. gün
])
@pytest.mark.parametrize("reason", ["damaged", "wrong_item"])
def test_r7_window_boundaries(delivered, expected, reason):
    r = evaluate(facts(delivered=delivered, estimated="2018-09-30"), reason, TODAY)
    assert r.decision is expected and r.rule == "R7"


def test_same_day_count_differs_by_reason():
    f = facts(delivered="2018-08-10", estimated="2018-09-30")  # 22. gün
    assert evaluate(f, "changed_mind", TODAY).decision is Decision.NOT_ELIGIBLE_WINDOW
    assert evaluate(f, "damaged", TODAY).decision is Decision.ELIGIBLE


# --------------------------------------------------------------------------- #
# R8 tutar
# --------------------------------------------------------------------------- #
TWO_ITEMS = ((100.0, 20.0, None), (50.5, 9.5, None))


def test_r8_r6_eligible_refunds_price_only():
    r = evaluate(facts(items=TWO_ITEMS), "changed_mind", TODAY)
    assert r.refund_amount == 150.5 and not r.freight_only


def test_r8_r7_eligible_includes_freight():
    r = evaluate(facts(items=TWO_ITEMS), "damaged", TODAY)
    assert r.refund_amount == 180.0


def test_r8_not_eligible_without_r9_is_zero_and_no_request():
    r = evaluate(facts(delivered="2018-08-01", estimated="2018-08-05"), "changed_mind", TODAY)
    assert r.refund_amount == 0 and r.approval_status is None and not r.can_open_request


def test_r8_other_decisions_are_zero():
    assert evaluate(facts(status="shipped", delivered=None), "damaged", TODAY).refund_amount == 0


# --------------------------------------------------------------------------- #
# R9 kargo bayrağı
# --------------------------------------------------------------------------- #
def test_r9_flag_boundary_10_vs_9_days():
    late10 = evaluate(facts(delivered="2018-08-25", estimated="2018-08-15"), "late", TODAY)
    late9 = evaluate(facts(delivered="2018-08-25", estimated="2018-08-16"), "late", TODAY)
    assert late10.freight_refund_flag and late10.late_days == 10
    assert not late9.freight_refund_flag and late9.late_days == 9


def test_r9_uses_calendar_days_not_timestamps():
    # 9 gün + 23 saat zaman damgası olarak <10 gün, takvim günü olarak 10 gün
    r = evaluate(facts(delivered="2018-08-25 23:00:00", estimated="2018-08-15 00:00:00"), "late", TODAY)
    assert r.freight_refund_flag


def test_r9_eligible_r6_adds_freight():
    r = evaluate(facts(items=TWO_ITEMS, delivered="2018-08-25", estimated="2018-08-10"),
                 "changed_mind", TODAY)
    assert r.decision is Decision.ELIGIBLE and r.refund_amount == 180.0


def test_r9_outside_window_within_30_days_is_freight_only():
    r = evaluate(facts(items=TWO_ITEMS, delivered="2018-08-10", estimated="2018-07-25"),
                 "changed_mind", TODAY)   # 22. gün, 16 gün geç
    assert r.decision is Decision.NOT_ELIGIBLE_WINDOW
    assert r.freight_only and r.refund_amount == 29.5 and r.can_open_request
    assert r.approval_status is ApprovalStatus.PENDING_REVIEW


def test_r9_after_30_days_no_request():
    r = evaluate(facts(delivered="2018-07-31", estimated="2018-07-01"), "late", TODAY)
    assert r.decision is Decision.NOT_ELIGIBLE_WINDOW and r.freight_refund_flag
    assert r.refund_amount == 0 and not r.can_open_request


# --------------------------------------------------------------------------- #
# R10 onay seviyesi
# --------------------------------------------------------------------------- #
def test_r10_exactly_500_is_review():
    r = evaluate(facts(items=((500.0, 30.0, None),)), "changed_mind", TODAY)
    assert r.refund_amount == 500.0 and r.approval_status is ApprovalStatus.PENDING_REVIEW


def test_r10_over_500_is_senior():
    r = evaluate(facts(items=((500.01, 0.0, None),)), "changed_mind", TODAY)
    assert r.approval_status is ApprovalStatus.PENDING_SENIOR


def test_r10_threshold_applies_after_freight():
    r = evaluate(facts(items=((480.0, 30.0, None),)), "damaged", TODAY)  # 510 kargo dahil
    assert r.refund_amount == 510.0 and r.approval_status is ApprovalStatus.PENDING_SENIOR


# --------------------------------------------------------------------------- #
# R11 tekrar talep
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", ["PENDING_REVIEW", "PENDING_SENIOR", "APPROVED", "REJECTED"])
def test_r11_existing_request_blocks(status):
    r = evaluate(facts(requests=(status,)), "damaged", TODAY)
    assert r.decision is Decision.DUPLICATE_REQUEST and r.refund_amount == 0


def test_r11_comes_after_status_checks():
    r = evaluate(facts(status="shipped", delivered=None, requests=("PENDING_REVIEW",)), "damaged", TODAY)
    assert r.decision is Decision.IN_TRANSIT


def test_r11_comes_before_window():
    r = evaluate(facts(delivered="2018-06-01", requests=("REJECTED",)), "damaged", TODAY)
    assert r.decision is Decision.DUPLICATE_REQUEST


# --------------------------------------------------------------------------- #
# R12 ödeme dağılımı
# --------------------------------------------------------------------------- #
def test_r12_voucher_share():
    r = evaluate(facts(items=((100.0, 20.0, None),), payment_total=120.0, voucher_total=30.0),
                 "changed_mind", TODAY)
    assert r.refund_amount == 100.0 and r.voucher_refund == 25.0 and r.original_method_refund == 75.0


def test_r12_no_payments_all_to_original_method():
    r = evaluate(facts(payment_total=0.0), "changed_mind", TODAY)
    assert r.voucher_refund == 0 and r.original_method_refund == r.refund_amount


# --------------------------------------------------------------------------- #
# R13 satıcı kaynaklı gecikme
# --------------------------------------------------------------------------- #
def test_r13_carrier_after_shipping_limit_flags_seller():
    r = evaluate(facts(carrier="2018-08-20 10:00:00",
                       items=((100.0, 20.0, "2018-08-18 00:00:00"),)), "late", TODAY)
    assert r.seller_caused_delay and "R13_SELLER_DELAY" in r.flags


def test_r13_same_day_is_not_flagged():
    r = evaluate(facts(carrier="2018-08-18 22:00:00",
                       items=((100.0, 20.0, "2018-08-18 09:00:00"),)), "late", TODAY)
    assert not r.seller_caused_delay


def test_r13_any_item_late_is_enough():
    r = evaluate(facts(carrier="2018-08-20",
                       items=((10.0, 1.0, "2018-08-25"), (10.0, 1.0, "2018-08-19"))), "late", TODAY)
    assert r.seller_caused_delay


def test_r13_does_not_change_decision():
    r = evaluate(facts(delivered="2018-08-01", carrier="2018-07-20",
                       items=((10.0, 1.0, "2018-07-10"),)), "changed_mind", TODAY)
    assert r.decision is Decision.NOT_ELIGIBLE_WINDOW and r.seller_caused_delay


# --------------------------------------------------------------------------- #
# Girdi doğrulama
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["", "refund", "DAMAGED!", None])
def test_invalid_reason_rejected(bad):
    with pytest.raises(ValueError, match="Geçersiz iade gerekçesi"):
        evaluate(facts(), bad, TODAY)


def test_reason_is_case_insensitive():
    assert evaluate(facts(), " Damaged ", TODAY).rule == "R7"


# --------------------------------------------------------------------------- #
# DB katmanı: R1 yetki SQL'de, oturum customer_unique_id ile
# --------------------------------------------------------------------------- #
def _hex(n: int) -> str:
    return f"{n:032x}"


@pytest.fixture
def con() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    for name, (_, cols, pk) in SCHEMA.items():
        pk_sql = f", PRIMARY KEY ({pk})" if pk else ""
        c.execute(f"CREATE TABLE {name} ({', '.join(f'{n} {t}' for n, t in cols)}{pk_sql})")
    c.executescript(RETURN_REQUESTS_DDL)
    # Ayşe (u-ayse): iki sipariş, iki farklı customer_id. Yabancı müşteri (u-other): 10 sipariş.
    rows = [(_hex(1), "c-a1", "u-ayse"), (_hex(2), "c-a2", "u-ayse")]
    rows += [(_hex(100 + i), f"c-x{i}", "u-other") for i in range(10)]
    for oid, cid, uid in rows:
        c.execute("INSERT INTO customers VALUES (?,?,?,?,?)", (cid, uid, "01000", "sp", "SP"))
        c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
                  (oid, cid, "delivered", "2018-08-01 10:00:00", None, "2018-08-03 09:00:00",
                   "2018-08-25 15:00:00", "2018-08-10 00:00:00"))
        c.execute("INSERT INTO order_items VALUES (?,?,?,?,?,?,?)",
                  (oid, 1, "p1", "s1", "2018-08-02 00:00:00", 300.0, 40.0))
        c.execute("INSERT INTO order_payments VALUES (?,?,?,?,?)", (oid, 1, "voucher", 1, 34.0))
        c.execute("INSERT INTO order_payments VALUES (?,?,?,?,?)", (oid, 2, "credit_card", 1, 306.0))
    yield c
    c.close()


def test_db_own_order_across_customer_ids(con):
    for oid in (_hex(1), _hex(2)):
        r = check_return_eligibility(con, oid, "u-ayse", "damaged", TODAY)
        assert r.decision is Decision.ELIGIBLE and r.refund_amount == 340.0


def test_db_facts_are_complete(con):
    f = load_order_facts(con, _hex(1), "u-ayse")
    assert f.status == "delivered" and f.delivered_date == date(2018, 8, 25)
    assert f.payment_total == 340.0 and f.voucher_total == 34.0
    assert f.carrier_date == date(2018, 8, 3) and f.items[0].shipping_limit_date == date(2018, 8, 2)


def test_db_unauthorized_access_100_percent_not_found(con):
    """G2: 10 yabancı order_id → %100 NOT_FOUND; var olmayanla aynı sonuç."""
    foreign = [_hex(100 + i) for i in range(10)]
    results = [check_return_eligibility(con, oid, "u-ayse", "damaged", TODAY) for oid in foreign]
    assert all(r.decision is Decision.NOT_FOUND for r in results)
    nonexistent = check_return_eligibility(con, _hex(999), "u-ayse", "damaged", TODAY)
    assert results[0].to_dict() == nonexistent.to_dict()  # varlık ifşa edilmez


@pytest.mark.parametrize("uid", ["", None, "u-ayse' OR '1'='1"])
def test_db_bad_session_identity_is_not_found(con, uid):
    assert check_return_eligibility(con, _hex(1), uid, "damaged", TODAY).decision is Decision.NOT_FOUND


def test_db_existing_request_triggers_r11(con):
    con.execute("INSERT INTO return_requests VALUES "
                "('rr1', ?, 'u-ayse', 'damaged', 340, 0, 'PENDING_REVIEW', '2018-09-01', 'mert')",
                (_hex(1),))
    assert check_return_eligibility(con, _hex(1), "u-ayse", "damaged", TODAY).decision \
        is Decision.DUPLICATE_REQUEST
    assert check_return_eligibility(con, _hex(2), "u-ayse", "damaged", TODAY).decision \
        is Decision.ELIGIBLE


def test_db_r12_and_r13_flags(con):
    r = check_return_eligibility(con, _hex(1), "u-ayse", "damaged", TODAY)
    assert r.voucher_refund == 34.0 and r.original_method_refund == 306.0
    assert r.seller_caused_delay  # kargo 08-03 > limit 08-02
    assert r.freight_refund_flag and r.late_days == 15  # 08-25 − 08-10
