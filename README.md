# ShopOps: Pazaryeri Müşteri Destek & Operasyon Ajanı

Müşterinin "siparişim nerede / iade edebilir miyim" sorusunu kurallara uygun ve yetkili biçimde cevaplayan, operasyon ekibinin metrik sorularını tanımlı metriklerden SQL üreterek yanıtlayan bir agent sistemi.

> **Durum:** M0, iskelet + veri katmanı. Demo linki, teknik yazı linki, mimari diyagram ve eval tablosu ilerleyen kilometre taşlarında eklenecek.

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

Gereksinim: Python 3.11+ (SQL doğrulayıcıdaki `SQLITE_LIMIT_ATTACHED` için).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # GEMINI_API_KEY'i doldur

python scripts/download_data.py                                   # yolu yazdırır
python scripts/load_sqlite.py --src <yukarıdaki_yol> --db data/shopops.db
python scripts/smoke.py --db data/shopops.db --src <yukarıdaki_yol>
pytest -q
```

`smoke.py` hem ham CSV satır sayılarını hem de kural/metrik kontrol değerlerini (R0 sonrası 99.421 sipariş, late_delivery_rate 0,0672, R6 penceresi 3.656 sipariş vb.) belgeyle karşılaştırır.

## Teknoloji seçimleri

| Katman | Seçim | Gerekçe |
|---|---|---|
| LLM | Gemini 3.5 Flash Lite (ücretsiz katman) | Function calling; ücretsiz katmanda en yüksek günlük limit (500 istek). Groq/Ollama'ya `.env`'de tek satırla geçilir. |
| Hakem | Gemini 3.8 Flash | Değerlendirilen modelden farklı ve daha güçlü |
| Embedding | `intfloat/multilingual-e5-small` | Yerel, kotasız, deploy RAM'ine sığar |
| Veri | SQLite | Tek dosya; salt okunur bağlantı ile güvenli SQL |
| Agent framework | *Karar bekliyor* | |

`src/llm_client.py` her çağrıyı önbelleğe alır (aynı istek kota harcamaz), dakikalık/günlük limitlere uyar, kota bitince kullanıcıya anlaşılır mesaj verir ve log'a yalnız token/gecikme yazar, içerik yazmaz (G6).

## Veri kaynağı ve lisans

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), **CC BY-NC-SA 4.0**: ticari kullanım yok, kaynak atfı zorunlu, türevler aynı lisansla. Ham veri ve ondan türetilmiş DB repoya konmaz; indirme ve yükleme script'leri konur.

## Repo yapısı

```
config.py              SIMULATED_TODAY, model adları, iş kuralı eşikleri (tek kaynak)
src/llm_client.py      önbellekli, hız sınırlı, sağlayıcıdan bağımsız LLM istemcisi
scripts/download_data.py
scripts/load_sqlite.py CSV → SQLite, R0 dışlama, indeksler, boş return_requests
scripts/smoke.py       belgedeki sayılarla karşılaştırma
tests/                 birim testleri
```

## Yapay zekâ desteği

Bu repoda kod yazımında yapay zekâ asistanı kullanıldı. Mimari kararlar, eval tasarımı ve hata analizi yazara aittir. [Ref: Genel Rehber §11]
