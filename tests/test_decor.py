"""Süs bileşeni: yapı, erişilebilirlik ve kapatma anahtarı."""
from __future__ import annotations

import re

from src import decor


def test_garland_structure():
    html = decor.garland_html()
    assert html.count('class="sd-bulb"') == decor.SEGMENTS * decor.BULBS_PER_SEGMENT
    assert "sd-hang left" in html and "sd-hang right" in html
    assert "\n\n" not in html and not re.search(r"^\s", html, re.M)  # markdown'da kod bloğu olmasın


def test_garland_is_accessible():
    html = decor.garland_html()
    assert 'aria-hidden="true"' in html and "pointer-events: none" in html
    assert "prefers-reduced-motion: reduce" in html
    durations = [float(d) for d in re.findall(r"animation-duration:([\d.]+)s", html)]
    assert durations and min(durations) >= 1.6          # yavaş yanıp sönme, hızlı flaş yok


def test_bulbs_sit_on_wire():
    ys = [float(y) for y in re.findall(r"top:([\d.]+)px", decor.garland_html())]
    assert all(decor.HOOK_Y <= y <= decor.HOOK_Y + decor.SAG for y in ys)
