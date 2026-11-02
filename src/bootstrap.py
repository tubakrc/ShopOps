"""Deploy için: DB yoksa Olist'i indirip SQLite'ı kurar. Ham veri ve DB repoya girmez.

[Ref: 02_ShopOps_Eticaret.pdf §8 Deploy notu "uygulama veriyi açılışta indirip DB'yi kendisi kursun";
 §3 lisans notu]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import config

logger = logging.getLogger("shopops.bootstrap")


def ensure_database(db_path: Path = config.DB_PATH) -> Path:
    """DB varsa dokunmaz; yoksa kagglehub ile indirir (hesap gerekmez) ve kurar (~1 dk)."""
    db_path = Path(db_path)
    if db_path.exists():
        return db_path
    sys.path.insert(0, str(config.ROOT / "scripts"))
    import kagglehub
    from load_sqlite import build_database

    logger.info("Veri indiriliyor: olistbr/brazilian-ecommerce")
    src = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    counts = build_database(src, db_path, with_geolocation=False)
    logger.info("DB kuruldu: %s sipariş", counts.get("orders"))
    return db_path
