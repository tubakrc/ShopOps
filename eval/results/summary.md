# Eval sonuçları

Model: `gemini-3.5-flash-lite` · temperature: 0.0 · tarih (simüle): 2018-09-01 · golden set: 33 örnek (support 16, policy 6, attack 11)

| Sistem | Task success | Tool-call accuracy | Karar doğruluğu | Uydurma karar oranı | Doğru kaynak (politika) | Policy faithfulness (judge) | Yetkisiz erişim engelleme | Saldırı/kapsam dışı direnci | Ort. adım | Ort. token | p50 gecikme (sn) | p95 gecikme (sn) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 tool'suz LLM | 69% | — | 38% | 62% | 100% | — | — | 100% | 1 | 3216 | 0.93 | 1.36 |
| B1 naif RAG | 54% | — | 15% | 69% | 100% | 100% (n=4) | — | 83% | 1 | 600 | 0.7 | 1.59 |
| B2 ham şema SQL | — | — | — | — | — | — | — | — | — | — | — | — |
| Final (agent) | 100% | 100% | 100% | 0% | 100% | 100% (n=4) | 100% | 100% | 2.15 | 3777 | 1.63 | 3.76 |

"—": bu sistem için anlamsız (B0/B1'in sipariş verisine ve tool'a erişimi yok) ya da henüz ölçülmedi (B2 ve operasyon grubu Ops Data Agent ile gelir). Gecikme = LLM + tool süresi; ücretsiz katman hız sınırlayıcısının beklemesi hariçtir. Faithfulness: hakem modelin (`gemini-3.8-flash`) desteklenen iddia oranı; n = hüküm verilen örnek.
