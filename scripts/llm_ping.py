"""LLM bağlantısını ve tool-calling gidiş-dönüşünü test eder (2–3 istek harcar).

Kullanım:
    python scripts/llm_ping.py
    python scripts/llm_ping.py --model gemini-3.8-flash

Kontrol edilen: anahtar ve model adı doğru mu; model tool çağırıyor mu; tool sonucu (ve Gemini'nin
tool_calls içindeki ek alanları) geri gönderildiğinde ikinci istek hatasız dönüyor mu.
Önbellek kapalıdır, her çalıştırma gerçek istek yapar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.llm_client import LLMClient, LLMError, RateLimiter, ResponseStore  # noqa: E402

TOOL = {"type": "function", "function": {
    "name": "get_order_status", "description": "Siparişin durumunu döndürür.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"]}}}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.LLM_MODEL)
    args = p.parse_args()

    store = ResponseStore(config.CACHE_PATH)
    try:
        llm = LLMClient(config.LLM_PROVIDER, args.model, 0.0, store,
                        RateLimiter(config.LLM_RPM, config.LLM_RPD, store, args.model),
                        use_cache=False)
    except LLMError as exc:
        print(f"✗ {exc}")
        return 1
    print(f"Sağlayıcı: {config.LLM_PROVIDER} · model: {args.model}")

    try:
        r = llm.chat([{"role": "user", "content": "Yalnızca 'merhaba' yaz."}])
        print(f"✓ Düz çağrı: {r.content!r} ({r.prompt_tokens}+{r.completion_tokens} token, {r.latency_s} sn)")

        msgs = [{"role": "user", "content": "abc123 numaralı siparişim nerede? Tool kullan."}]
        r1 = llm.chat(msgs, tools=[TOOL])
        if not r1.tool_calls:
            print(f"✗ Model tool çağırmadı. Cevap: {r1.content!r}")
            return 1
        tc = r1.tool_calls[0]
        print(f"✓ Tool çağrısı: {tc.name}({tc.arguments})")
        msgs += [r1.assistant_message(),
                 {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                  "content": json.dumps({"status": "shipped", "estimated_delivery_date": "2018-09-12"})}]
        r2 = llm.chat(msgs, tools=[TOOL])
        print(f"✓ Tool sonucu sonrası cevap: {r2.content!r}")
    except LLMError as exc:
        cause = exc.__cause__
        print(f"✗ {exc}\n  Ayrıntı: {type(cause).__name__}: {str(cause)[:300]}")
        print("  Model adı yanlışsa AI Studio → model → 'Get code' çıktısındaki kimliği .env'e yaz.")
        return 1
    print("\nTÜMÜ TAMAM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
