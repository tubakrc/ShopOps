"""Sağlayıcıdan bağımsız, önbellekli ve hız sınırlı LLM istemcisi.

Gemini, Groq ve Ollama'nın üçü de OpenAI uyumlu bir chat-completions endpoint'i sunar;
sağlayıcı değiştirmek .env'de tek satırdır. [Ref: Genel Rehber Ek A]

Neden bu katman var:
- Ücretsiz katman kotası küçük (Flash Lite: 15 RPM, 500 RPD). Aynı istek önbellekten
  döner, kota yemez. [Ref: ShopOps §11 "Ücretsiz katman limitleri"]
- Her çağrının token/gecikme bilgisi döner; agent katmanı bunu adım logu olarak yazar.
  [Ref: Genel Rehber §3 #7 Temel: "adımların ve token sayısının loglanması"]
- Log'a mesaj içeriği yazılmaz; yalnız model, token, gecikme, önbellek anahtarının
  kısa hash'i. [Ref: ShopOps §6.2 G6; Genel Rehber §8 "Log'lar da veridir"]

Not: Günlük kota sayacı gerçek UTC gününü kullanır, çünkü sağlayıcının kotası gerçek
zamana göre sıfırlanır. İş kuralları ise yalnız config.SIMULATED_TODAY kullanır.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("shopops.llm")

PROVIDERS: dict[str, dict[str, Optional[str]]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None},
}

QUOTA_MESSAGE = (
    "Ücretsiz LLM kotası şu an dolu. Lütfen birkaç dakika sonra tekrar deneyin "
    "ya da .env içinde LLM_PROVIDER'ı groq/ollama yapın."
)


class LLMError(RuntimeError):
    """Kurtarılamayan istemci hatası (eksik anahtar, bilinmeyen sağlayıcı, API hatası)."""


class QuotaExhausted(LLMError):
    """Dakikalık/günlük kota doldu. Mesajı kullanıcıya gösterilebilir. [Ref: Genel Rehber §3 #9]"""


@dataclass(frozen=True)
class ToolCall:
    """Modelin istediği tek bir tool çağrısı. arguments JSON çözülemezse None olur."""

    id: str
    name: str
    arguments: Optional[dict[str, Any]]
    raw_arguments: str


@dataclass
class LLMResponse:
    """Normalize edilmiş model cevabı + gözlemlenebilirlik alanları."""

    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cache_hit: bool = False
    model: str = ""
    # Asistan mesajındaki tool_calls, sağlayıcının döndürdüğü hâliyle (None alanlar atılmış).
    # Konuşmaya aynen geri eklenir: Gemini 3 modelleri tool_calls içindeki ek alanları
    # (thought signature) bir sonraki istekte geri bekler.
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def assistant_message(self) -> dict[str, Any]:
        """Konuşma geçmişine eklenecek asistan mesajı."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.raw_tool_calls:
            msg["tool_calls"] = self.raw_tool_calls
        return msg

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------- #
# Önbellek + günlük sayaç
# --------------------------------------------------------------------------- #
class ResponseStore:
    """SQLite tabanlı cevap önbelleği ve günlük istek sayacı."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Streamlit her yeniden çalıştırmayı farklı bir thread'de yapar; bağlantı paylaşılır, kilitle korunur.
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS daily_usage (
                day TEXT NOT NULL, model TEXT NOT NULL, requests INTEGER NOT NULL,
                PRIMARY KEY (day, model));
            """
        )

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._con.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict[str, Any]) -> None:
        with self._lock, self._con:
            self._con.execute(
                "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
                (key, json.dumps(response)),
            )

    def requests_today(self, day: str, model: str) -> int:
        with self._lock:
            row = self._con.execute(
                "SELECT requests FROM daily_usage WHERE day = ? AND model = ?", (day, model)
            ).fetchone()
        return int(row[0]) if row else 0

    def increment(self, day: str, model: str) -> None:
        with self._lock, self._con:
            self._con.execute(
                """INSERT INTO daily_usage (day, model, requests) VALUES (?, ?, 1)
                   ON CONFLICT(day, model) DO UPDATE SET requests = requests + 1""",
                (day, model),
            )


# --------------------------------------------------------------------------- #
# Hız sınırlayıcı
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Dakikalık sınır için çağrılar arasına eşit aralık koyar; günlük sınırda durdurur.

    clock/sleep enjekte edilebilir, böylece testler gerçekten beklemez.
    """

    def __init__(
        self,
        rpm: int,
        rpd: int,
        store: ResponseStore,
        model: str,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm <= 0 or rpd <= 0:
            raise ValueError("rpm ve rpd pozitif olmalı")
        self.min_interval = 60.0 / rpm
        self.rpd = rpd
        self.store = store
        self.model = model
        self.clock = clock
        self.sleep = sleep
        self._last_call: Optional[float] = None

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def acquire(self) -> None:
        """Gerekirse bekler; günlük kota dolduysa QuotaExhausted fırlatır."""
        if self.store.requests_today(self._today(), self.model) >= self.rpd:
            raise QuotaExhausted(QUOTA_MESSAGE)
        if self._last_call is not None:
            wait = self.min_interval - (self.clock() - self._last_call)
            if wait > 0:
                self.sleep(wait)
        self._last_call = self.clock()
        self.store.increment(self._today(), self.model)


# --------------------------------------------------------------------------- #
# İstemci
# --------------------------------------------------------------------------- #
Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _is_rate_limit_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or "429" in str(exc)


def _is_transient_error(exc: Exception) -> bool:
    """Sağlayıcı tarafı geçici hata (500/502/503/504, InternalServerError)."""
    code = getattr(exc, "status_code", None)
    return (isinstance(code, int) and code >= 500) or type(exc).__name__ in {
        "InternalServerError", "APITimeoutError", "APIConnectionError"}


def _openai_transport(provider: str) -> Transport:
    """OpenAI SDK ile gerçek HTTP çağrısı yapan transport'u kurar."""
    if provider not in PROVIDERS:
        raise LLMError(f"Bilinmeyen LLM_PROVIDER: {provider!r}. Seçenekler: {sorted(PROVIDERS)}")
    spec = PROVIDERS[provider]
    key_env = spec["key_env"]
    api_key = os.getenv(key_env) if key_env else "ollama"
    if not api_key:
        raise LLMError(f"{key_env} tanımlı değil. .env.example dosyasını .env olarak kopyalayıp doldurun.")

    from openai import OpenAI  # yalnız gerçek çağrıda gerekli

    client = OpenAI(base_url=spec["base_url"], api_key=api_key, max_retries=0)

    def call(request: dict[str, Any]) -> dict[str, Any]:
        return client.chat.completions.create(**request).model_dump()

    return call


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_drop_none(v) for v in obj]
    return obj


def _parse_response(raw: dict[str, Any]) -> tuple[Optional[str], list[ToolCall], int, int]:
    message = raw["choices"][0]["message"]
    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        raw_args = tc["function"].get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args)
            args = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            args = None  # agent katmanı modele "argümanlar geçersiz JSON" geri bildirimi verir
        calls.append(ToolCall(id=tc.get("id") or "", name=tc["function"]["name"],
                              arguments=args, raw_arguments=raw_args))
    usage = raw.get("usage") or {}
    return (message.get("content"), calls,
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))


class LLMClient:
    """Tek giriş noktası: chat(messages, tools) -> LLMResponse."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        store: Optional[ResponseStore] = None,
        limiter: Optional[RateLimiter] = None,
        transport: Optional[Transport] = None,
        max_retries_on_429: int = 2,
        backoff_s: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.store = store
        self.limiter = limiter
        self.transport = transport or _openai_transport(provider)
        self.max_retries_on_429 = max_retries_on_429
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.use_cache = use_cache and store is not None
        # refresh_cache: önbellekten okuma, ama yeni cevabı yaz (temiz gecikme ölçümü için)
        self.refresh_cache = refresh_cache and store is not None

    @classmethod
    def from_config(cls, model: Optional[str] = None) -> "LLMClient":
        """config.py ve .env'den varsayılan istemciyi kurar."""
        import config

        model_name = model or config.LLM_MODEL
        store = ResponseStore(config.CACHE_PATH)
        limiter = RateLimiter(config.LLM_RPM, config.LLM_RPD, store, model_name)
        return cls(config.LLM_PROVIDER, model_name, config.LLM_TEMPERATURE, store, limiter)

    def _cache_key(self, request: dict[str, Any]) -> str:
        payload = json.dumps({"provider": self.provider, **request}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat(
        self, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None
    ) -> LLMResponse:
        """Modeli çağırır. Önbellekte varsa kota harcamadan döner."""
        request: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": self.temperature,
        }
        if tools:
            request["tools"] = tools

        key = self._cache_key(request)
        if self.use_cache and not self.refresh_cache:
            cached = self.store.get(key)  # type: ignore[union-attr]
            if cached is not None:
                # Önbellek, ilk çağrının gerçek gecikmesini de saklar; raporlar 0 sn görmez.
                resp = self._to_response(cached, latency=float(cached.get("_latency_s", 0.0)),
                                         cache_hit=True)
                self._log(key, resp)
                return resp

        raw, latency = self._call_with_retry(request)
        resp = self._to_response(raw, latency=latency, cache_hit=False)
        if self.use_cache or self.refresh_cache:
            self.store.put(key, {**raw, "_latency_s": round(latency, 3)})  # type: ignore[union-attr]
        self._log(key, resp)
        return resp

    def _call_with_retry(self, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
        for attempt in range(self.max_retries_on_429 + 1):
            if self.limiter:
                self.limiter.acquire()
            started = time.perf_counter()
            try:
                raw = self.transport(request)
                return raw, time.perf_counter() - started
            except Exception as exc:  # sağlayıcı SDK'ları farklı istisna tipleri fırlatır
                transient = _is_transient_error(exc)
                if not (_is_rate_limit_error(exc) or transient):
                    raise LLMError(f"LLM çağrısı başarısız: {type(exc).__name__}") from exc
                if attempt == self.max_retries_on_429:
                    if transient:
                        raise LLMError(f"LLM çağrısı başarısız: {type(exc).__name__}") from exc
                    raise QuotaExhausted(QUOTA_MESSAGE) from exc
                self.sleep(self.backoff_s * (attempt + 1))
        raise AssertionError("unreachable")

    def _to_response(self, raw: dict[str, Any], latency: float, cache_hit: bool) -> LLMResponse:
        content, calls, p_tok, c_tok = _parse_response(raw)
        raw_calls = _drop_none(raw["choices"][0]["message"].get("tool_calls") or [])
        return LLMResponse(content=content, tool_calls=calls, prompt_tokens=p_tok,
                           completion_tokens=c_tok, latency_s=round(latency, 3),
                           cache_hit=cache_hit, model=self.model, raw_tool_calls=raw_calls)

    def _log(self, key: str, resp: LLMResponse) -> None:
        # G6: içerik yok; yalnız ölçüm alanları ve anahtarın kısa hash'i.
        logger.info(json.dumps({
            "event": "llm_call", "key": key[:10], "model": resp.model,
            "cache_hit": resp.cache_hit, "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens, "latency_s": resp.latency_s,
            "tool_calls": [tc.name for tc in resp.tool_calls],
        }))


__all__ = ["LLMClient", "LLMResponse", "ToolCall", "ResponseStore", "RateLimiter",
           "LLMError", "QuotaExhausted", "QUOTA_MESSAGE"]
