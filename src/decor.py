"""Arayüz süsü: ekranın üstünde iple asılı, yanıp sönen ampuller + balkabağı ve noel ağacı.

Saf HTML/CSS (JS yok). `SHOPOPS_DECOR=off` ile kapanır (config.DECOR).
Erişilebilirlik: ampuller yumuşak ve yavaş yanıp söner (≥1,6 sn döngü; ışığa duyarlılık için hızlı
flaş yok); işletim sisteminde "hareketi azalt" açıksa tüm animasyon durur. Süs tıklamaları engellemez
(pointer-events: none) ve ekran okuyuculardan gizlidir (aria-hidden).
"""
from __future__ import annotations

# Marka turuncusu önde; sıcak bir "holiday" paleti
BULB_COLORS = ["#F94915", "#F5B700", "#2E8B57", "#D62839", "#FFF1C1"]
SEGMENTS = 7          # kancalar arası sarkma sayısı
BULBS_PER_SEGMENT = 3
HOOK_Y = 6            # px: telin kancadaki yüksekliği
SAG = 22              # px: segment ortasında telin sarkması
# Balkabağı ve ağaç uçlarda değil, 1. ve 5. kancada asılı: sağ üstteki menü düğmelerinin üstüne binmesin.

PUMPKIN_SVG = """<svg viewBox="0 0 64 60" width="40" height="38" aria-hidden="true">
  <path d="M31 10 C30 4 34 1 38 2" stroke="#4B5D2A" stroke-width="3" fill="none" stroke-linecap="round"/>
  <ellipse cx="18" cy="35" rx="14" ry="20" fill="#E8590C"/>
  <ellipse cx="46" cy="35" rx="14" ry="20" fill="#E8590C"/>
  <ellipse cx="32" cy="35" rx="15" ry="22" fill="#F76707"/>
  <path d="M24 31 l5 -5 l5 5 z M38 31 l5 -5 l5 5 z" fill="#3B1F0B" transform="translate(-5 0)"/>
  <path d="M20 42 q12 9 24 0 l-3 4 l-3 -3 l-3 4 l-3 -4 l-3 4 l-3 -4 l-3 3 z" fill="#3B1F0B"/>
</svg>"""

TREE_SVG = """<svg viewBox="0 0 60 72" width="38" height="46" aria-hidden="true">
  <polygon points="30,4 32.5,10 39,10 34,14 36,20 30,16.5 24,20 26,14 21,10 27.5,10" fill="#F5B700"/>
  <polygon points="30,14 46,36 14,36" fill="#2E8B57"/>
  <polygon points="30,26 50,52 10,52" fill="#2B7A4B"/>
  <polygon points="30,38 54,66 6,66" fill="#256B42"/>
  <rect x="26" y="66" width="8" height="6" fill="#7A4A21"/>
  <circle cx="24" cy="44" r="2.4" fill="#D62839"/><circle cx="37" cy="50" r="2.4" fill="#F94915"/>
  <circle cx="20" cy="60" r="2.4" fill="#F5B700"/><circle cx="40" cy="61" r="2.4" fill="#D62839"/>
  <circle cx="31" cy="30" r="2.2" fill="#F94915"/>
</svg>"""


def _wire_path() -> str:
    """viewBox genişliği 1000; her segment bir kuadratik sarkma."""
    w = 1000 / SEGMENTS
    parts = [f"M0,{HOOK_Y}"]
    for i in range(SEGMENTS):
        x0 = i * w
        parts.append(f"Q{x0 + w / 2:.1f},{HOOK_Y + 2 * SAG} {x0 + w:.1f},{HOOK_Y}")
    return " ".join(parts)


def _bulbs() -> list[str]:
    """Ampuller telin üzerinde: x yüzde, y piksel (tel viewBox'ı yatayda esner, dikeyde sabit)."""
    out, k = [], 0
    for i in range(SEGMENTS):
        for j in range(1, BULBS_PER_SEGMENT + 1):
            t = j / (BULBS_PER_SEGMENT + 1)
            x_pct = (i + t) * 100 / SEGMENTS
            y = HOOK_Y + 2 * SAG * 2 * t * (1 - t)      # kuadratik eğrinin y'si (P1 = 2·SAG)
            color = BULB_COLORS[k % len(BULB_COLORS)]
            delay = (k * 0.37) % 2.4
            dur = 1.6 + (k % 4) * 0.35
            tilt = (-12, 0, 12)[(j - 1) % 3]
            out.append(
                f'<span class="sd-bulb" style="left:{x_pct:.2f}%;top:{y:.1f}px;--c:{color};'
                f'animation-delay:{delay:.2f}s;animation-duration:{dur:.2f}s;'
                f'transform:translateX(-50%) rotate({tilt}deg)"></span>')
            k += 1
    return out


def garland_html() -> str:
    """st.markdown(..., unsafe_allow_html=True) ile sayfaya eklenecek HTML + CSS.

    Boş ve girintili satırlar atılır: markdown, HTML içindeki boş satırdan sonraki 4 boşluk girintili
    satırları kod bloğu sanar.
    """
    return "\n".join(line.strip() for line in _raw_html().splitlines() if line.strip())


def _raw_html() -> str:
    return f"""
<style>
.sd-garland {{ position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 999991;
  pointer-events: none; }}
.sd-wire {{ position: absolute; top: 0; left: 0; width: 100%; height: 64px; }}
.sd-bulb {{ position: absolute; width: 10px; height: 15px; margin-top: 2px;
  border-radius: 50% 50% 50% 50% / 40% 40% 60% 60%; background: var(--c);
  box-shadow: 0 0 6px 2px var(--c); animation-name: sd-twinkle; animation-iteration-count: infinite;
  animation-timing-function: ease-in-out; transform-origin: 50% -4px; }}
.sd-bulb::before {{ content: ""; position: absolute; top: -4px; left: 2px; width: 6px; height: 5px;
  border-radius: 1px; background: #3F3A33; }}
@keyframes sd-twinkle {{ 0%, 100% {{ opacity: 1; box-shadow: 0 0 8px 3px var(--c); }}
  50% {{ opacity: .45; box-shadow: 0 0 2px 0 var(--c); }} }}
.sd-hang {{ position: absolute; top: {HOOK_Y}px; transform-origin: 50% 0;
  animation: sd-swing 3.2s ease-in-out infinite; }}
.sd-hang::before {{ content: ""; display: block; width: 1.5px; height: 10px; margin: 0 auto;
  background: #3F3A33; }}
.sd-hang.left {{ left: calc({100 * 1 / SEGMENTS:.3f}% - 20px); }}
.sd-hang.right {{ left: calc({100 * 5 / SEGMENTS:.3f}% - 19px); animation-delay: -1.6s; }}
@keyframes sd-swing {{ 0%, 100% {{ transform: rotate(-6deg); }} 50% {{ transform: rotate(6deg); }} }}
@media (prefers-reduced-motion: reduce) {{
  .sd-bulb, .sd-hang {{ animation: none !important; }}
}}
</style>
<div class="sd-garland" aria-hidden="true">
  <svg class="sd-wire" viewBox="0 0 1000 64" preserveAspectRatio="none">
    <path d="{_wire_path()}" fill="none" stroke="#3F3A33" stroke-width="1.6"
          vector-effect="non-scaling-stroke"/>
  </svg>
  {''.join(_bulbs())}
  <div class="sd-hang left">{PUMPKIN_SVG}</div>
  <div class="sd-hang right">{TREE_SVG}</div>
</div>
"""
