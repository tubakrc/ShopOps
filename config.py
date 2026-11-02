"""Merkezi yapılandırma: tarih, yollar, model adları, iş kuralı eşikleri.

Tek doğruluk kaynağıdır; kodun hiçbir yerinde datetime.now() ile "bugün" hesaplanmaz.
[Ref: 02_ShopOps_Eticaret.pdf §1, §11 "Makine saati kullanma"; Genel Rehber §5 config.py]
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Simülasyon zamanı [Ref: ShopOps §1] -------------------------------------
SIMULATED_TODAY: date = date(2018, 9, 1)

# --- Yollar ------------------------------------------------------------------
DB_PATH: Path = Path(os.getenv("SHOPOPS_DB", ROOT / "data" / "shopops.db"))
CACHE_PATH: Path = Path(os.getenv("LLM_CACHE_DB", ROOT / ".cache" / "llm_cache.db"))
POLICIES_DIR: Path = ROOT / "policies"
POLICY_INDEX_PATH: Path = ROOT / "index" / "policy_index.npz"  # repoya girer (deploy için)
TRACE_PATH: Path = Path(os.getenv("SHOPOPS_TRACE", ROOT / "logs" / "traces.jsonl"))
ESCALATION_PATH: Path = ROOT / "logs" / "escalations.jsonl"

# --- LLM [Ref: Genel Rehber Ek A; AI Studio Rate Limit ekranı, 15 Ekim 2026] --
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # gemini | groq | ollama
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

# Ücretsiz katman limitleri (Gemini 3.5 Flash Lite: 15 RPM / 500 RPD).
# Güvenlik payı için varsayılanlar limitin biraz altında.
LLM_RPM: int = int(os.getenv("LLM_RPM", "14"))
LLM_RPD: int = int(os.getenv("LLM_RPD", "480"))
# Hakem modeli (Gemini 3.8 Flash ücretsiz katman: 5 RPM / 20 RPD)
JUDGE_RPM: int = int(os.getenv("JUDGE_RPM", "4"))
JUDGE_RPD: int = int(os.getenv("JUDGE_RPD", "19"))

# [Ref: ShopOps §8 Adım 4]. Deploy'da bellek yetmezse "trigram-hash-4096" (bağımlılıksız yedek).
EMBED_MODEL: str = os.getenv("SHOPOPS_EMBEDDER", "intfloat/multilingual-e5-small")

# Herkese açık demoda bir oturumun harcayabileceği en fazla mesaj (ortak ücretsiz kotayı korur)
DEMO_MAX_MESSAGES_PER_SESSION: int = int(os.getenv("DEMO_MAX_MESSAGES", "15"))

# --- Agent döngüsü koruması [Ref: ShopOps §6.2 G10] ---------------------------
MAX_STEPS: int = 8
MAX_SAME_CALL: int = 2

# --- İş kuralı eşikleri [Ref: ShopOps §6.1] ----------------------------------
RETURN_WINDOW_DAYS_STANDARD: int = 14   # R6: changed_mind, late
RETURN_WINDOW_DAYS_EXTENDED: int = 30   # R7: damaged, wrong_item; R9 kargo talebi
FREIGHT_REFUND_LATE_DAYS: int = 10      # R9
SENIOR_APPROVAL_THRESHOLD_BRL: float = 500.0  # R10

# --- SQL doğrulayıcı [Ref: ShopOps §6.2 G3, G4] ------------------------------
SQL_ROW_LIMIT: int = 1000
SQL_TIMEOUT_SECONDS: float = 5.0
SQL_MAX_SELF_CORRECTIONS: int = 2
