"""PII'siz adım/token/gecikme izleme (trace). [Ref: ShopOps §6.2 G6; Genel Rehber §3 #7]

Kural: trace'e serbest metin (müşteri mesajı, model cevabı, tool çıktısı) yazılmaz. Kimlikler
sha256(id)[:10] ile yazılır. Test kriteri: trace dökümünde 32 karakterlik hex id regex'i 0 eşleşme.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

RAW_ID_RE = re.compile(r"\b[0-9a-f]{32}\b")

# Gemini ücretsiz katman fiyatı 0; ücretli eşdeğer maliyeti raporlamak için .env'den verilebilir.
DEFAULT_PRICE_PER_MTOK = {"input": 0.0, "output": 0.0}


def hash_id(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]


def redact_ids(text: str) -> str:
    """Metindeki 32 karakterlik hex id'leri hash'leriyle değiştirir (eskalasyon özeti vb.)."""
    return RAW_ID_RE.sub(lambda m: f"id:{hash_id(m.group(0))}", text)


def safe_args(args: Optional[dict]) -> dict:
    """Tool argümanlarını trace'e uygun hâle getirir: id'ler hash, serbest metin yalnız uzunluk."""
    if not isinstance(args, dict):
        return {"_invalid": True}
    out: dict[str, Any] = {}
    for key, val in args.items():
        if isinstance(val, (int, float, bool)) or val is None:
            out[key] = val
        elif key.endswith("_id") or (isinstance(val, str) and RAW_ID_RE.fullmatch(val.strip())):
            out[key] = f"h:{hash_id(str(val).strip())}"
        elif key in {"reason", "priority", "status"} and isinstance(val, str) and len(val) <= 20:
            out[key] = val  # enum değerleri
        else:
            out[key] = f"<{len(str(val))} karakter>"
    return out


@dataclass
class Tracer:
    """Bir agent turunun adımlarını toplar ve JSONL'e ekler."""

    path: Optional[Path]
    session_hash: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[dict] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)

    def llm_step(self, step: int, model: str, prompt_tokens: int, completion_tokens: int,
                 latency_s: float, cache_hit: bool, tool_names: list[str]) -> None:
        self._add({"type": "llm", "step": step, "model": model, "prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens, "latency_s": latency_s,
                   "cache_hit": cache_hit, "tool_calls": tool_names})

    def tool_step(self, step: int, name: str, args: Optional[dict], latency_s: float,
                  result_code: str) -> None:
        self._add({"type": "tool", "step": step, "tool": name, "args": safe_args(args),
                   "latency_s": round(latency_s, 4), "result": result_code})

    def finish(self, stopped_reason: str) -> dict:
        llm = [e for e in self.events if e["type"] == "llm"]
        summary = {
            "type": "summary", "steps": len(llm),
            "tool_calls": sum(e["type"] == "tool" for e in self.events),
            "prompt_tokens": sum(e["prompt_tokens"] for e in llm),
            "completion_tokens": sum(e["completion_tokens"] for e in llm),
            "llm_latency_s": round(sum(e["latency_s"] for e in llm), 3),
            "total_latency_s": round(time.perf_counter() - self._started, 3),
            "stopped_reason": stopped_reason,
        }
        self._add(summary)
        self._flush()
        return summary

    def _add(self, event: dict) -> None:
        self.events.append({"run_id": self.run_id, "session": self.session_hash, **event})

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for e in self.events:
                # G6 son savunma: bir alan gözden kaçsa bile ham id dosyaya yazılmaz.
                f.write(redact_ids(json.dumps(e, ensure_ascii=False)) + "\n")
