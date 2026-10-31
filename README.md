# ShopOps: Pazaryeri Müşteri Destek & Operasyon Ajanı

Müşterinin "siparişim nerede / iade edebilir miyim" sorusunu kurallara uygun ve yetkili biçimde cevaplayan, operasyon ekibinin metrik sorularını tanımlı metriklerden SQL üreterek yanıtlayan bir agent sistemi.

> **Durum:** Veri katmanı, kural motoru (R0–R13), politika RAG'i, Support Agent ve HITL'li CLI tamam. Demo linki, teknik yazı linki, mimari diyagram ve eval tablosu ilerleyen kilometre taşlarında eklenecek.

## Problem ve hedef kullanıcı

Senaryo kurgusaldır: "Örnekpazar" adlı çok satıcılı bir pazaryeri. Geçmiş sipariş verisi olarak Olist'in herkese açık Brezilya e-ticaret veri seti kullanılır. Veri 2018'de bittiği için sistemde "bugün" sabittir: `SIMULATED_TODAY = 2018-09-01`.

| Persona | Kim | Bugünkü acı |
|---|---|---|
| Ayşe | Müşteri | "Siparişim nerede?" için kuyrukta bekliyor |
| Mert | Destek temsilcisi | Taleplerin çoğu durum ve iade; her seferinde politikayı elle kontrol ediyor |
| Selin | Operasyon yöneticisi | "Geçen ay SP'de geç teslim oranı neydi?" için 2–3 gün ticket bekliyor |

**Maliyet varsayımı (kurgusal):** Ayda 20.000 talep, ~%60'ı durum/iade tekrarı, talep başı 6 dk → ~1.200 temsilci saati/ay. Operasyonda haftada ~15 metrik ticket'ı, her biri ~2 gün bekliyor.

**Başarı ölçütleri** (hedefler, ShopOps §9.2): task success ≥ %80, tool-call accuracy ≥ %90, karar kodu doğruluğu %100, SQL execution accuracy ≥ %75, yetkisiz erişim engelleme %100.

## Neden agent?

- Müşteri cümlesi serbest: "kulaklık gelmedi, parayı geri istiyorum" hem durum sorgusu hem iade hem belki iptal; tool sırası cümleye göre değişir.
- Operasyon soruları metrik × eyalet × kategori × dönem kombinasyonlarıdır; sabit rapor yetmez.
- Buna karşılık iade uygunluğu Python kural motoruyla, yetki kodla, yazma işlemi insan onayıyla çözülür. Projenin özü LLM'in serbest olduğu yer ile kodla kilitlenen yeri ayırmaktır.

Bu iddia baseline ölçümüyle (B0 tool'suz LLM, B1 naif RAG, B2 ham şema SQL) gösterilecek.

## 10 örnek kullanıcı sorusu

| # | Kim | Soru | Beklenen davranış |
|---|---|---|---|
| 1 | Müşteri | Son siparişim nerede? | `list_customer_orders` → `get_order_status` |
| 2 | Müşteri | `<order_id>` siparişimi iade etmek istiyorum, beden olmadı. | `check_return_eligibility` → ELIGIBLE → özet + onay → `create_return_request` |
| 3 | Müşteri | Ürün 3 hafta önce geldi, iade olur mu? | NOT_ELIGIBLE_WINDOW |
| 4 | Müşteri | Ürün çok geç geldi, kargo paramı geri alabilir miyim? | R9 gecikme kontrolü → kargo iadesi var/yok |
| 5 | Müşteri | Siparişimle ilgili sorun var. | Netleştirici soru; tahminle tool çağırma yok |
| 6 | Müşteri | `<başkasının order_id'si>` nerede? | NOT_FOUND |
| 7 | Ops | 2018 Temmuz'da SP eyaletinde geç teslim oranı neydi? | `run_metric("late_delivery_rate", …)` |
| 8 | Ops | En az 100 siparişi olan kategoriler arasında geç teslim oranı en yüksek 5'i | Semantic layer'dan SQL, HAVING, LIMIT 5 |
| 9 | Ops | Geç teslim yorum puanını düşürüyor mu? | İlişkiyi göster, nedensellik iddia etme |
| 10 | Herkes | Bana Python'da bir oyun yaz. | Kibar ret |

## Kurulum

Gereksinim: Python 3.12 ve [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env            # GEMINI_API_KEY'i doldur (Windows: copy .env.example .env)

uv run python scripts/download_data.py                            # yolu yazdırır
uv run python scripts/load_sqlite.py --src <yukarıdaki_yol> --db data/shopops.db
uv run python scripts/smoke.py --db data/shopops.db --src <yukarıdaki_yol>
uv run python scripts/verify_rules.py --db data/shopops.db
uv run python scripts/build_policy_index.py --compare   # e5 indeksini kurar, retrieval'ı ölçer
uv run python scripts/llm_ping.py                       # anahtar + tool-calling gidiş-dönüş testi
uv run pytest -q

uv run python cli.py --demo --steps                     # müşteri destek demosu
```

`smoke.py` ham CSV satır sayılarını ve veri kontrol değerlerini (R0 sonrası 99.421 sipariş, late_delivery_rate 0,0672, R6 penceresi 3.656 sipariş vb.) belgeyle karşılaştırır. `verify_rules.py` kural motorunu 99.421 siparişin tamamında çalıştırır ve §9.1'deki kontrol değerlerini (130 / 352, R9 dağılımı 33 / 28 / 2.195) doğrular.

## Teknoloji seçimleri

| Katman | Seçim | Gerekçe |
|---|---|---|
| LLM | Gemini 3.5 Flash Lite (ücretsiz katman) | Function calling; ücretsiz katmanda en yüksek günlük limit (500 istek). Groq/Ollama'ya `.env`'de tek satırla geçilir. |
| Hakem | Gemini 3.8 Flash | Değerlendirilen modelden farklı ve daha güçlü |
| Embedding | `intfloat/multilingual-e5-small` | Yerel, kotasız, deploy RAM'ine sığar |
| Veri | SQLite | Tek dosya; salt okunur bağlantı ile güvenli SQL |
| Agent framework | Yok: açık tool-calling döngüsü (`src/agents/support_agent.py`) | Döngü ~100 satır. Yetki, tutar, onay ve döngü sınırları kodda ve birim testiyle kanıtlı; framework'ün gizli retry/prompt davranışı yok. 6 günlük takvimde öğrenme maliyeti de yok. |
| Vektör deposu | Yok: numpy kosinüs, tek `.npz` dosyası | 30 chunk için vektör DB gereksiz; indeks repoda durur, deploy'da yeniden kurulmaz. |

`src/llm_client.py` her çağrıyı önbelleğe alır (aynı istek kota harcamaz), dakikalık/günlük limitlere uyar, kota bitince kullanıcıya anlaşılır mesaj verir ve log'a yalnız token/gecikme yazar, içerik yazmaz (G6).

## Mimari (Temel seviye)

```
Kullanıcı ─▶ CLI (oturum = customer_unique_id) ─▶ Support Agent (Gemini, tool-calling döngüsü)
                                                     │  max 8 adım · aynı çağrı ≤ 2
                                                     ▼
                         ┌──────────── Tool katmanı (kimlik oturumdan, Pydantic doğrulama) ────────────┐
                         │ list_customer_orders · get_order_status        → SQLite (mode=ro)          │
                         │ check_return_eligibility                       → returns.py (R0–R13)        │
                         │ search_policies                                → politika indeksi (e5)      │
                         │ escalate_to_human                              → iç kuyruk (log)            │
                         │ create_return_request → APPROVAL_REQUIRED (yazmaz)                          │
                         └─────────────────────────────────────────────────────────────────────────────┘
CLI "Onaylıyor musunuz? [e/h]" ─▶ commit_pending() ─▶ kuralları yeniden uygula ─▶ return_requests (tek yazma)
```

### Tool'lar

| Tool | Tür | Not |
|---|---|---|
| `list_customer_orders` | okuma | Oturum müşterisinin tüm `customer_id`'leri üzerinden, yeni → eski |
| `get_order_status` | okuma | Başkasının siparişi ve var olmayan sipariş aynı NOT_FOUND cevabı |
| `check_return_eligibility` | okuma | Karar ve tutarı LLM değil, kural motoru verir |
| `search_policies` | okuma | Doküman adı + bölüm döner; cevapta kaynak gösterilir |
| `escalate_to_human` | iç kuyruk | Kayıt değiştirmez; özetteki id'ler hash'lenir |
| `create_return_request` | yazma (HITL) | Yalnız teklif hazırlar, tutar kural motoruyla eşleşmezse reddeder |

Yazma, LLM'in çağıramadığı `commit_pending()` fonksiyonundadır. Onay, sunucu tarafındaki oturumda duran teklifin kimliğiyle eşleşir; model bu teklifi üretemez, "onayladım" demesi de bir şey değiştirmez.

### Politika dokümanları ve retrieval

`policies/` altında 7 kurgusal Örnekpazar politika dokümanı var; R0–R13'ün her biri en az bir dokümanda geçer. Chunk birimi `## ` bölümüdür (30 chunk): bölümler tek konuya odaklı, sabit token penceresi ise bir kuralı ikiye bölebiliyor. Retrieval kalitesi `eval/policy_retrieval.jsonl` (15 sorgu) üzerinde hit@1/hit@3/MRR ile ölçülür; sonuç `eval/results/retrieval.csv`. e5 modeli yüklenemezse (ör. deploy RAM'i) sistem bağımlılıksız trigram embedder'a düşer ve bunu loglar.

### Gözlemlenebilirlik

Her tur `logs/traces.jsonl`'a adım adım yazılır: LLM adımı başına token ve gecikme, tool adı, hash'lenmiş argümanlar, sonuç kodu. Mesaj ve cevap metni yazılmaz; 32 karakterlik ham id'ler yazılmadan önce `sha256[:10]` ile değiştirilir (G6).

## Veri kaynağı ve lisans

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), **CC BY-NC-SA 4.0**: ticari kullanım yok, kaynak atfı zorunlu, türevler aynı lisansla. Ham veri ve ondan türetilmiş DB repoya konmaz; indirme ve yükleme script'leri konur.

## Repo yapısı

```
config.py              SIMULATED_TODAY, model adları, iş kuralı eşikleri (tek kaynak)
src/llm_client.py      önbellekli, hız sınırlı, sağlayıcıdan bağımsız LLM istemcisi
src/returns.py         R0–R13 iade/iptal kural motoru (LLM'siz, deterministik)
src/tools/support.py   6 tool + HITL onay kapısı (commit_pending)
src/agents/support_agent.py  tool-calling döngüsü, döngü koruması
src/knowledge/policy_search.py  bölüm bazlı chunking, e5 / trigram embedding, arama
src/observability.py   PII'siz trace (token, gecikme, hash'li id)
policies/              7 politika dokümanı
index/policy_index.npz politika indeksi (repoda)
cli.py                 müşteri destek CLI'ı ("Onaylıyor musunuz? [e/h]")
scripts/download_data.py
scripts/load_sqlite.py CSV → SQLite, R0 dışlama, indeksler, boş return_requests
scripts/smoke.py       belgedeki sayılarla karşılaştırma
scripts/verify_rules.py kural motorunun tüm veride doğrulanması
scripts/build_policy_index.py  indeksi kurar, retrieval'ı ölçer
scripts/llm_ping.py    LLM bağlantı ve tool-calling testi
eval/policy_retrieval.jsonl    retrieval değerlendirme sorguları
tests/                 birim testleri
```

## Yapay zekâ desteği

Bu repoda kod yazımında yapay zekâ asistanı kullanıldı. Mimari kararlar, eval tasarımı ve hata analizi yazara aittir. [Ref: Genel Rehber §11]
