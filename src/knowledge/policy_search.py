"""Politika dokümanları için retrieval: bölüm bazlı chunking + yoğun embedding + kosinüs.

[Ref: 02_ShopOps_Eticaret.pdf §8 Adım 4; Genel Rehber §3 #3]

Kararlar (README'de gerekçeli):
- Chunk = bir "## " bölümü. Politika dokümanları kısa ve bölümleri tek konuya odaklı; sabit
  token penceresi bir kuralı ikiye bölebiliyor. Her chunk doküman başlığını da taşır.
- Vektör DB yok: ~30 chunk için numpy kosinüs yeterli, indeks tek .npz dosyası olarak deploy'a girer.
- Embedder değiştirilebilir. Varsayılan intfloat/multilingual-e5-small ("query:"/"passage:"
  önekleriyle). Model yüklenemezse (deploy RAM'i, ağ yok) bağımlılıksız TrigramEmbedder'a düşer
  ve bunu loglar; demo çökmez. İndeks hangi embedder'la kurulduysa sorgu da onunla yapılır.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger("shopops.rag")


@dataclass(frozen=True)
class Chunk:
    doc: str        # dosya adı, ör. "04_hasarli_yanlis_urun.md"
    title: str      # doküman başlığı
    section: str    # bölüm başlığı
    text: str       # bölüm gövdesi

    @property
    def embed_text(self) -> str:
        return f"{self.title} — {self.section}\n{self.text}"


@dataclass(frozen=True)
class Hit:
    doc: str
    title: str
    section: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks(policies_dir: Path) -> list[Chunk]:
    """Her .md dosyasını '## ' bölümlerine böler. Başlıksız giriş paragrafı 'Genel' bölümüdür."""
    chunks: list[Chunk] = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((l[2:].strip() for l in lines if l.startswith("# ")), path.stem)
        section, body = "Genel", []

        def flush() -> None:
            text = "\n".join(body).strip()
            if text:
                chunks.append(Chunk(path.name, title, section, text))

        for line in lines:
            if line.startswith("# "):
                continue
            if line.startswith("## "):
                flush()
                section, body = line[3:].strip(), []
            else:
                body.append(line)
        flush()
    if not chunks:
        raise FileNotFoundError(f"{policies_dir} içinde politika dokümanı yok")
    return chunks


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    payload = json.dumps([asdict(c) for c in chunks], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Embedder'lar
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], kind: str) -> np.ndarray:  # kind: "query" | "passage"
        ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class E5Embedder:
    """intfloat/multilingual-e5-small; ilk kullanımda yüklenir (~470 MB indirme, bir kez)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prefixed = [f"{kind}: {t}" for t in texts]
        vecs = self._load().encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def _fold(text: str) -> str:
    """Küçük harf + Türkçe/aksan katlama: 'Hasarlı' ve 'hasarli' aynı trigramları üretir."""
    text = text.replace("I", "ı").replace("İ", "i").lower()
    text = text.replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


class TrigramEmbedder:
    """Bağımlılıksız yedek: karakter 3-gram'larının hash'lenmiş sayımı (lexical).

    Anlamsal benzerlik yakalamaz; e5 yüklenemezse demo çalışsın ve testler ağsız koşsun diye var.
    """

    name = "trigram-hash-4096"

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", _fold(text)):
                padded = f" {word} "
                for i in range(len(padded) - 2):
                    out[row, zlib.crc32(padded[i:i + 3].encode()) % self.dim] += 1.0
        return _normalize(out)


def default_embedder(name: str) -> Embedder:
    return TrigramEmbedder() if name == TrigramEmbedder.name else E5Embedder(name)


# --------------------------------------------------------------------------- #
# İndeks
# --------------------------------------------------------------------------- #
class PolicyIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk ve vektör sayısı eşleşmiyor")
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.fingerprint = corpus_fingerprint(chunks)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "PolicyIndex":
        vecs = embedder.encode([c.embed_text for c in chunks], kind="passage")
        return cls(chunks, np.asarray(vecs, dtype=np.float32), embedder)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"embedder": self.embedder.name, "fingerprint": self.fingerprint,
                "chunks": [asdict(c) for c in self.chunks]}
        np.savez_compressed(path, vectors=self.vectors, meta=np.array(json.dumps(meta, ensure_ascii=False)))

    @classmethod
    def load(cls, path: Path, embedder: Optional[Embedder] = None) -> "PolicyIndex":
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        emb = embedder or default_embedder(meta["embedder"])
        if emb.name != meta["embedder"]:
            raise ValueError(f"İndeks {meta['embedder']} ile kurulmuş; sorgu embedder'ı {emb.name}")
        return cls([Chunk(**c) for c in meta["chunks"]], data["vectors"], emb)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        query = (query or "").strip()
        if not query:
            return []
        q = self.embedder.encode([query], kind="query")[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[: max(1, min(k, len(self.chunks)))]
        return [Hit(self.chunks[i].doc, self.chunks[i].title, self.chunks[i].section,
                    self.chunks[i].text, round(float(scores[i]), 4)) for i in order]


def open_index(index_path: Path, policies_dir: Path, embedder_name: str) -> PolicyIndex:
    """Kayıtlı indeksi açar; yoksa, dokümanlar değiştiyse ya da e5 yüklenemezse yeniden kurar.

    e5 yüklenemezse TrigramEmbedder'a düşer ve uyarır (demo çökmez).
    """
    chunks = load_chunks(policies_dir)
    try:
        idx = PolicyIndex.load(index_path)
        if idx.fingerprint == corpus_fingerprint(chunks) and idx.embedder.name == embedder_name:
            idx.search("test", k=1)  # embedder gerçekten yükleniyor mu?
            return idx
        logger.info("Politika indeksi eski; yeniden kuruluyor.")
    except FileNotFoundError:
        logger.info("Politika indeksi yok; kuruluyor.")
    except Exception as exc:  # model indirilemedi / bozuk dosya
        logger.warning("İndeks açılamadı (%s); yeniden kuruluyor.", type(exc).__name__)

    try:
        idx = PolicyIndex.build(chunks, default_embedder(embedder_name))
    except Exception as exc:
        logger.warning("%s yüklenemedi (%s); TrigramEmbedder'a düşülüyor.", embedder_name,
                       type(exc).__name__)
        idx = PolicyIndex.build(chunks, TrigramEmbedder())
    try:
        idx.save(index_path)
    except OSError:
        pass  # salt okunur deploy diski: bellekte devam
    return idx
