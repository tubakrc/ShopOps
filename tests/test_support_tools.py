"""Support tool'ları: yetki (G2), HITL (G1), şema doğrulama, PII'siz çıktı (G6), RAG.

[Ref: 02_ShopOps_Eticaret.pdf §5.3, §6.2, §9.4 A1–A3]
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

import config
from src.knowledge.policy_search import PolicyIndex, TrigramEmbedder, load_chunks
from src.returns import NOT_FOUND_MESSAGE
from src.tools.support import (SUPPORT_TOOLS, Session, ToolContext, commit_pending, execute_tool)
from support_fixtures import (AYSE, FOREIGN, NONEXISTENT, O_BIG, O_ELIG, O_FREIGHT, O_FUTURE,
                              O_SHIP, build_support_db)


@pytest.fixture(scope="module")
def policy_index() -> PolicyIndex:
    return PolicyIndex.build(load_chunks(config.POLICIES_DIR), TrigramEmbedder())


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return build_support_db(tmp_path / "support.db")


@pytest.fixture
def ctx(db: Path, policy_index, tmp_path: Path) -> ToolContext:
    c = ToolContext(Session("customer", AYSE), db, policy_index,
                    escalation_path=tmp_path / "esc.jsonl")
    yield c
    c.close()


def run(ctx, name, **args):
    return execute_tool(ctx, name, args)[0]


def count_requests(db: Path) -> int:
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM return_requests").fetchone()[0]
    con.close()
    return n


# --------------------------------------------------------------------------- #
# G2: yetki tool katmanında
# --------------------------------------------------------------------------- #
def test_no_tool_accepts_customer_identity():
    for spec in SUPPORT_TOOLS:
        props = spec.openai_schema()["function"]["parameters"].get("properties", {})
        assert not any("customer" in p for p in props), spec.name


@pytest.mark.parametrize("tool,extra", [
    ("get_order_status", {}),
    ("check_return_eligibility", {"reason": "damaged"}),
    ("create_return_request", {"reason": "damaged", "amount": 110.0}),
])
def test_foreign_orders_100_percent_not_found(ctx, tool, extra):
    results = [run(ctx, tool, order_id=oid, **extra) for oid in FOREIGN]
    assert all(r["error"] == "NOT_FOUND" for r in results)
    assert results[0] == run(ctx, tool, order_id=NONEXISTENT, **extra)  # varlık ifşa yok
    assert results[0]["message"] == NOT_FOUND_MESSAGE


def test_injected_customer_id_argument_is_rejected(ctx):
    # A2: model argümana başka kimlik eklerse şema reddeder; oturum kimliği değişmez.
    r = run(ctx, "get_order_status", order_id=FOREIGN[0], customer_unique_id="u-other")
    assert r["error"] == "INVALID_ARGUMENTS"


def test_ops_session_cannot_use_customer_tools(db, policy_index):
    ops = ToolContext(Session("ops"), db, policy_index)
    assert run(ops, "list_customer_orders")["error"] == "FORBIDDEN"
    assert run(ops, "get_order_status", order_id=O_ELIG)["error"] == "FORBIDDEN"
    ops.close()


def test_list_orders_spans_customer_ids_newest_first(ctx):
    r = run(ctx, "list_customer_orders", limit=10)
    ids = [o["order_id"] for o in r["orders"]]
    assert set(ids) == {O_ELIG, O_BIG, O_SHIP, O_FUTURE, O_FREIGHT}
    dates = [o["purchase_date"] for o in r["orders"]]
    assert dates == sorted(dates, reverse=True)
    assert run(ctx, "list_customer_orders", limit=2)["count"] == 2


def test_list_orders_status_filter(ctx):
    r = run(ctx, "list_customer_orders", status="shipped")
    assert {o["order_id"] for o in r["orders"]} == {O_SHIP, O_FUTURE}  # gelecekte teslim = yolda


# --------------------------------------------------------------------------- #
# get_order_status
# --------------------------------------------------------------------------- #
def test_order_status_has_no_pii_fields(ctx):
    r = run(ctx, "get_order_status", order_id=O_ELIG)
    assert r["status"] == "delivered" and r["delivered_date"] == "2018-08-25"
    assert r["items"][0]["category"] == "furniture_decor"
    text = json.dumps(r)
    assert "sao paulo" not in text and "01000" not in text and "customer" not in text


def test_future_delivery_is_shown_as_in_transit(ctx):
    r = run(ctx, "get_order_status", order_id=O_FUTURE)
    assert r["status"] == "shipped" and r["delivered_date"] is None


# --------------------------------------------------------------------------- #
# Doğrulama ve hata geri bildirimi
# --------------------------------------------------------------------------- #
def test_invalid_reason_is_reported_not_guessed(ctx):
    r = run(ctx, "check_return_eligibility", order_id=O_ELIG, reason="refund")
    assert r["error"] == "INVALID_ARGUMENTS" and r["details"][0]["field"] == "reason"


def test_invalid_json_and_unknown_tool(ctx):
    assert execute_tool(ctx, "get_order_status", None)[0]["error"] == "INVALID_JSON"
    r = execute_tool(ctx, "run_sql", {"query": "SELECT 1"})[0]   # G5: müşteride run_sql yok
    assert r["error"] == "UNKNOWN_TOOL"


def test_eligibility_payload(ctx):
    r = run(ctx, "check_return_eligibility", order_id=O_ELIG, reason="damaged")
    assert r["decision"] == "ELIGIBLE" and r["refund_amount"] == 340.0
    assert r["policy_doc"] == "01_iade_politikasi.md" and r["can_open_request"]


# --------------------------------------------------------------------------- #
# G1: HITL'siz yazma yok
# --------------------------------------------------------------------------- #
def test_create_return_request_never_writes(ctx, db):
    r = run(ctx, "create_return_request", order_id=O_ELIG, reason="damaged", amount=340.0)
    assert r["status"] == "APPROVAL_REQUIRED" and r["proposal"]["queue"] == "PENDING_REVIEW"
    assert count_requests(db) == 0


def test_amount_is_decided_by_rule_engine(ctx, db):
    r = run(ctx, "create_return_request", order_id=O_ELIG, reason="damaged", amount=9999.0)
    assert r["error"] == "AMOUNT_MISMATCH" and r["correct_amount"] == 340.0
    assert ctx.pending is None and count_requests(db) == 0


def test_not_eligible_order_gets_no_proposal(ctx):
    r = run(ctx, "create_return_request", order_id=O_SHIP, reason="damaged", amount=90.0)
    assert r["error"] == "NOT_ELIGIBLE" and r["decision"] == "IN_TRANSIT" and ctx.pending is None


def test_freight_only_proposal(ctx):
    r = run(ctx, "create_return_request", order_id=O_FREIGHT, reason="changed_mind", amount=25.5)
    assert r["status"] == "APPROVAL_REQUIRED" and r["proposal"]["freight_only"]


def test_commit_after_human_approval_writes_one_row(ctx, db):
    r = run(ctx, "create_return_request", order_id=O_BIG, reason="damaged", amount=620.0)
    out = commit_pending(ctx, r["proposal"]["proposal_id"], approved_by="customer-cli")
    assert out["status"] == "PENDING_SENIOR" and out["refund_amount_brl"] == 620.0
    con = sqlite3.connect(db)
    rows = con.execute("SELECT order_id, status, refund_amount, approved_by FROM return_requests").fetchall()
    con.close()
    assert rows == [(O_BIG, "PENDING_SENIOR", 620.0, "customer-cli")]
    # R11: ikinci talep artık mümkün değil
    again = run(ctx, "check_return_eligibility", order_id=O_BIG, reason="damaged")
    assert again["decision"] == "DUPLICATE_REQUEST"


def test_commit_requires_matching_proposal_and_approver(ctx, db):
    r = run(ctx, "create_return_request", order_id=O_ELIG, reason="damaged", amount=340.0)
    assert commit_pending(ctx, "fake123", "x")["error"] == "NO_PENDING_REQUEST"
    assert commit_pending(ctx, r["proposal"]["proposal_id"], "  ")["error"] == "APPROVAL_REQUIRED"
    assert count_requests(db) == 0


def test_read_connection_is_read_only(ctx):
    with pytest.raises(sqlite3.OperationalError):
        ctx.ro.execute("DELETE FROM orders")


# --------------------------------------------------------------------------- #
# search_policies ve escalate_to_human
# --------------------------------------------------------------------------- #
def test_search_policies_returns_sources(ctx):
    r = run(ctx, "search_policies", query="hasarlı ürün", k=3)
    docs = [h["doc"] for h in r["results"]]
    assert "04_hasarli_yanlis_urun.md" in docs and all(h["section"] for h in r["results"])


def test_escalation_redacts_ids(ctx, tmp_path):
    r = run(ctx, "escalate_to_human", summary=f"{O_ELIG} teslim tarihi eksik", priority="high")
    assert r["ticket"].startswith("ESC-")
    log = (tmp_path / "esc.jsonl").read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{32}\b", log) and AYSE not in log
