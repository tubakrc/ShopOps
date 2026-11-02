"""Streamlit arayüzü: AppTest ile uçtan uca (sahte LLM, sentetik DB).

Kanıtlanan: oturum kenar çubuğundan kurulur; iade önerisi onay kartı gösterir; tablo yalnız
"Onayla" düğmesiyle yazılır (G1); ara adımlar görünür; mesaj sınırı çalışır.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402
from src import scenarios  # noqa: E402
from test_eval import H, build_eval_db  # noqa: E402
from test_support_agent import ScriptedLLM, call, text_resp, tool_resp  # noqa: E402

APP = str(config.ROOT / "app.py")
ORDER = H(10)  # test_eval DB'sinde r6_eligible: 100 + 10 kargo


@pytest.fixture
def env(tmp_path, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()   # paylaşılan kaynaklar (DB yolu, indeks) testler arasında taşınmasın
    st.cache_data.clear()
    db = build_eval_db(tmp_path / "app.db")
    con = sqlite3.connect(db)
    con.execute("DELETE FROM return_requests")
    con.commit()
    con.close()
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(config, "POLICY_INDEX_PATH", tmp_path / "idx.npz")
    monkeypatch.setattr(config, "EMBED_MODEL", "trigram-hash-4096")
    monkeypatch.setattr(config, "ESCALATION_PATH", tmp_path / "esc.jsonl")
    monkeypatch.setattr(config, "TRACE_PATH", tmp_path / "trace.jsonl")
    import src.agents.support_agent as sa
    monkeypatch.setattr(sa.SupportAgent.__init__, "__defaults__",
                        (tmp_path / "trace.jsonl",) + sa.SupportAgent.__init__.__defaults__[1:])
    return db


def make_app(llm) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["llm_override"] = llm
    return at


def rows(db: Path) -> list:
    con = sqlite3.connect(db)
    out = con.execute("SELECT order_id, status, refund_amount FROM return_requests").fetchall()
    con.close()
    return out


def return_flow_llm():
    llm = ScriptedLLM([
        tool_resp(call("check_return_eligibility", {"order_id": ORDER, "reason": "changed_mind"})),
        tool_resp(call("create_return_request", {"order_id": ORDER, "reason": "changed_mind", "amount": 100.0})),
        text_resp("100,00 BRL iade talebi hazır; lütfen onaylayın."),
    ])
    llm.model = "fake"
    return llm


def test_app_starts_with_session_and_orders(env):
    at = make_app(ScriptedLLM([]))
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox[0].value == scenarios.DEMO_SCENARIOS[0][0]
    assert len(at.sidebar.code) >= 1          # müşterinin kendi sipariş numaraları
    assert any(b.label == "Oturumu yenile" for b in at.sidebar.button)
    assert at.chat_input[0] is not None


def test_return_flow_requires_button(env):
    at = make_app(return_flow_llm())
    at.run()
    at.chat_input[0].set_value(f"{ORDER} iade etmek istiyorum, beden olmadı").run()
    assert not at.exception
    assert any("Onayla" in b.label for b in at.button)
    assert rows(env) == []                                    # öneri var, yazma yok
    assert any("İade tutarı" == m.label for m in at.metric)
    expander_labels = [e.label for e in at.expander]
    assert any("2 tool" in label for label in expander_labels)  # ara adımlar görünüyor

    next(b for b in at.button if "Onayla" in b.label).click().run()
    assert rows(env) == [(ORDER, "PENDING_REVIEW", 100.0)]
    assert any("oluşturuldu" in m.markdown[0].value for m in at.chat_message if m.markdown)


def test_reject_writes_nothing(env):
    at = make_app(return_flow_llm())
    at.run()
    at.chat_input[0].set_value(f"{ORDER} iade").run()
    next(b for b in at.button if b.label == "Vazgeç").click().run()
    assert rows(env) == []
    assert not any("Onayla" in b.label for b in at.button)


def test_message_limit(env, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MAX_MESSAGES_PER_SESSION", 1)
    llm = ScriptedLLM([text_resp("Merhaba!"), text_resp("ikinci")])
    llm.model = "fake"
    at = make_app(llm)
    at.run()
    at.chat_input[0].set_value("selam").run()
    at.chat_input[0].set_value("tekrar").run()
    texts = [m.markdown[0].value for m in at.chat_message if m.markdown]
    assert any("mesaj sınırına" in t for t in texts) and len(llm.seen) == 1
