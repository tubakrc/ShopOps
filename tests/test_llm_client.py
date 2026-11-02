"""llm_client testleri: önbellek, hız sınırı, günlük kota, 429 yeniden deneme, PII'siz log.

Gerçek API çağrısı yok; transport, saat ve sleep enjekte edilir.
"""
from __future__ import annotations

import json
import logging
import re

import pytest

from src.llm_client import (LLMClient, LLMError, QuotaExhausted, RateLimiter, ResponseStore)


def fake_raw(content="Merhaba", tool_calls=None, p=12, c=3):
    return {"choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": p, "completion_tokens": c}}


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class CountingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class RateLimitErr(Exception):
    status_code = 429


def make_client(transport, rpm=60, rpd=100, clock=None, **kw):
    clock = clock or FakeClock()
    store = ResponseStore(":memory:")
    limiter = RateLimiter(rpm, rpd, store, "m", clock=clock, sleep=clock.sleep)
    return LLMClient("gemini", "m", 0.0, store, limiter, transport, sleep=clock.sleep, **kw), clock


MSG = [{"role": "user", "content": "Siparişim nerede?"}]


def test_parses_content_and_usage():
    client, _ = make_client(CountingTransport([fake_raw()]))
    resp = client.chat(MSG)
    assert resp.content == "Merhaba" and resp.total_tokens == 15 and not resp.cache_hit


def test_parses_tool_calls_and_flags_bad_json():
    calls = [
        {"id": "1", "function": {"name": "get_order_status", "arguments": '{"order_id": "abc"}'}},
        {"id": "2", "function": {"name": "list_customer_orders", "arguments": "{bozuk"}},
    ]
    client, _ = make_client(CountingTransport([fake_raw(None, calls)]))
    resp = client.chat(MSG, tools=[{"type": "function", "function": {"name": "x"}}])
    assert resp.tool_calls[0].arguments == {"order_id": "abc"}
    assert resp.tool_calls[1].arguments is None and resp.tool_calls[1].raw_arguments == "{bozuk"


def test_cache_hit_skips_transport_and_quota():
    t = CountingTransport([fake_raw()])
    client, _ = make_client(t)
    client.chat(MSG)
    second = client.chat(MSG)
    assert t.calls == 1 and second.cache_hit and second.content == "Merhaba"


def test_different_tools_or_model_are_different_cache_keys():
    t = CountingTransport([fake_raw()])
    client, _ = make_client(t)
    client.chat(MSG)
    client.chat(MSG, tools=[{"type": "function", "function": {"name": "x"}}])
    assert t.calls == 2


def test_rate_limiter_spaces_calls():
    clock = FakeClock()
    client, _ = make_client(CountingTransport([fake_raw()]), rpm=15, clock=clock, use_cache=False)
    client.chat(MSG)
    client.chat(MSG)
    assert clock.sleeps == [pytest.approx(4.0)]  # 60 / 15


def test_daily_quota_raises_friendly_error():
    client, _ = make_client(CountingTransport([fake_raw()]), rpd=2, use_cache=False)
    client.chat(MSG)
    client.chat(MSG)
    with pytest.raises(QuotaExhausted, match="kota"):
        client.chat(MSG)


def test_429_is_retried_then_succeeds():
    t = CountingTransport([RateLimitErr("429 Too Many Requests"), fake_raw("tamam")])
    client, clock = make_client(t, backoff_s=20)
    assert client.chat(MSG).content == "tamam"
    assert t.calls == 2 and 20 in clock.sleeps


def test_persistent_429_becomes_quota_exhausted():
    client, _ = make_client(CountingTransport([RateLimitErr("429")]), max_retries_on_429=1)
    with pytest.raises(QuotaExhausted):
        client.chat(MSG)


def test_other_errors_are_wrapped_without_leaking_message():
    client, _ = make_client(CountingTransport([ValueError("api key=SECRET123")]))
    with pytest.raises(LLMError) as exc:
        client.chat(MSG)
    assert "SECRET123" not in str(exc.value)


def test_missing_api_key_is_clear(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLMClient("gemini", "m")


def test_unknown_provider_rejected():
    with pytest.raises(LLMError, match="Bilinmeyen"):
        LLMClient("openrouter", "m")


def test_log_has_no_content_or_raw_ids(caplog):
    order_id = "e481f51cbdc54678b7cc49136f2d6af7"  # 32 karakterlik Olist kimliği biçimi
    client, _ = make_client(CountingTransport([fake_raw(f"{order_id} teslim edildi")]))
    with caplog.at_level(logging.INFO, logger="shopops.llm"):
        client.chat([{"role": "user", "content": f"{order_id} nerede?"}])
    text = caplog.text
    assert order_id not in text and "nerede" not in text
    assert not re.search(r"\b[0-9a-f]{32}\b", text)  # G6 test kriteri
    record = json.loads(caplog.records[0].getMessage())
    assert record["prompt_tokens"] == 12 and "latency_s" in record


def test_cache_hit_reports_original_latency():
    t = CountingTransport([fake_raw()])
    client, _ = make_client(t)
    first = client.chat(MSG)
    second = client.chat(MSG)
    assert second.cache_hit and second.latency_s == first.latency_s


def test_refresh_cache_calls_again_and_overwrites():
    t = CountingTransport([fake_raw("eski"), fake_raw("yeni")])
    client, clock = make_client(t)
    client.chat(MSG)
    clock2 = FakeClock()
    store = client.store
    refresher = LLMClient("gemini", "m", 0.0, store, RateLimiter(60, 100, store, "m", clock=clock2,
                          sleep=clock2.sleep), t, refresh_cache=True)
    assert refresher.chat(MSG).content == "yeni" and t.calls == 2
    assert client.chat(MSG).content == "yeni" and t.calls == 2  # normal istemci yeni kaydı okur


class ServerErr(Exception):
    status_code = 500


def test_transient_500_is_retried_once():
    t = CountingTransport([ServerErr("Internal error"), fake_raw("tamam")])
    client, clock = make_client(t, backoff_s=5)
    assert client.chat(MSG).content == "tamam" and t.calls == 2


def test_persistent_500_is_llm_error_not_quota():
    client, _ = make_client(CountingTransport([ServerErr("Internal error")]), max_retries_on_429=1)
    with pytest.raises(LLMError) as exc:
        client.chat(MSG)
    assert not isinstance(exc.value, QuotaExhausted)
