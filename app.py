"""ShopOps müşteri destek demosu (Streamlit).

    streamlit run app.py

Arayüz ilkeleri [Ref: ShopOps §7 Hedef; Genel Rehber §3 #8, §8]:
- Oturum (kimlik) kenar çubuğunda seçilir; sohbette söylenen hiçbir şey oturumu değiştirmez.
- İade talebi yalnız "Onayla" düğmesiyle yazılır (HITL). Model bu düğmeye erişemez.
- Her cevabın altında ara adımlar görünür: hangi tool, hangi sonuç, token ve süre.
- Ücretsiz kota bitince anlaşılır mesaj; oturum başına mesaj sınırı ortak kotayı korur.
"""
from __future__ import annotations

import os

import streamlit as st

# Streamlit Cloud'da sırlar st.secrets'ta; config import edilmeden önce ortama aktarılır.
try:
    for _key in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "SHOPOPS_EMBEDDER",
                 "DEMO_MAX_MESSAGES"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # secrets.toml yoksa (yerel çalışma) .env kullanılır
    pass

import sqlite3  # noqa: E402
from typing import Optional  # noqa: E402

import config  # noqa: E402
from src.agents.support_agent import SupportAgent  # noqa: E402
from src.bootstrap import ensure_database  # noqa: E402
from src.knowledge.policy_search import open_index  # noqa: E402
from src.llm_client import LLMClient, LLMError  # noqa: E402
from src.observability import hash_id  # noqa: E402
from src.scenarios import DEMO_SCENARIOS, resolve_customer  # noqa: E402
from src.tools.support import Session, ToolContext, commit_pending, reject_pending, result_code  # noqa: E402

st.set_page_config(page_title="ShopOps · Örnekpazar Destek", page_icon="🛍️", layout="wide")

# Marka renkleri. Palet .streamlit/config.toml'da da var; ama o dosya yalnız sunucu açılırken ve
# çalışma klasöründe bulunursa okunur, tarayıcıda seçili bir tema da onu ezebilir. Bu CSS her durumda
# uygulanır: ana alan #F9F8F4, kenar çubuğu bir ton koyusu, tüm düğmeler dolu turuncu.
BRAND_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
  background-color: #F9F8F4 !important;
}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
  background-color: #EFEDE6 !important;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  color: #2B2A26;
}
.stButton > button, [data-testid^="stBaseButton"] {
  background-color: #F94915 !important; color: #F9F8F4 !important;
  border: 1px solid #F94915 !important; font-weight: 600;
}
.stButton > button:hover, [data-testid^="stBaseButton"]:hover {
  background-color: #DE3D0D !important; border-color: #DE3D0D !important;
}
.stButton > button p, [data-testid^="stBaseButton"] p { color: inherit !important; }
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button p { color: #EFEDE6 !important; }
[data-testid="stChatInputSubmitButton"] { background-color: #F94915 !important; color: #F9F8F4 !important; }
</style>
"""
st.markdown(BRAND_CSS, unsafe_allow_html=True)

REASON_TR = {"changed_mind": "fikir değişikliği", "damaged": "hasarlı ürün",
             "wrong_item": "yanlış ürün", "late": "geç teslim"}
STATUS_TR = {"delivered": "teslim edildi", "shipped": "kargoda", "canceled": "iptal",
             "unavailable": "karşılanamadı", "processing": "hazırlanıyor", "invoiced": "faturalandı",
             "approved": "onaylandı", "created": "oluşturuldu"}
EXAMPLES = ["Son siparişim nerede?", "Ürün kırık geldi, iade etmek istiyorum.",
            "Hasarlı ürünü kaç gün içinde iade edebilirim?", "Bana Python'da bir oyun yaz."]


# --------------------------------------------------------------------------- #
# Paylaşılan kaynaklar (tüm ziyaretçiler)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Veri hazırlanıyor (ilk açılışta ~1 dk)…")
def get_db() -> str:
    return str(ensure_database(config.DB_PATH))


@st.cache_resource(show_spinner="Politika indeksi yükleniyor…")
def get_index():
    return open_index(config.POLICY_INDEX_PATH, config.POLICIES_DIR, config.EMBED_MODEL)


@st.cache_resource
def get_llm() -> Optional[LLMClient]:
    """Tek istemci: hız sınırlayıcı ve günlük sayaç tüm ziyaretçiler için ortak."""
    try:
        return LLMClient.from_config()
    except LLMError:
        return None


@st.cache_data
def scenario_customers(db: str) -> list[tuple[str, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for label, selector in DEMO_SCENARIOS:
        uid = resolve_customer(con, selector, config.SIMULATED_TODAY.isoformat())
        if uid:
            out.append((label, uid))
    con.close()
    return out


def customer_orders(db: str, uid: str) -> list[dict]:
    """Kenar çubuğu için oturum müşterisinin kendi siparişleri (yalnız kendi verisi)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT o.order_id, o.order_status, o.order_purchase_timestamp, o.order_delivered_customer_date "
        "FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.customer_unique_id = ? ORDER BY o.order_purchase_timestamp DESC LIMIT 10", (uid,)).fetchall()
    con.close()
    today = config.SIMULATED_TODAY.isoformat()
    out = []
    for oid, status, purchased, delivered in rows:
        if status == "delivered" and delivered and delivered[:10] > today:
            status = "shipped"  # R3: gelecekte teslim = henüz teslim edilmedi
        out.append({"Sipariş no": oid, "Durum": STATUS_TR.get(status, status),
                    "Sipariş tarihi": purchased[:10]})
    return out


# --------------------------------------------------------------------------- #
# Oturum durumu
# --------------------------------------------------------------------------- #
def start_session(uid: str, db: str) -> None:
    old: Optional[SupportAgent] = st.session_state.get("agent")
    if old is not None:
        old.ctx.close()
    llm = st.session_state.get("llm_override") or get_llm()
    ctx = ToolContext(Session("customer", uid), db, get_index(),
                      escalation_path=config.ESCALATION_PATH)
    st.session_state.agent = SupportAgent(llm, ctx) if llm is not None else None
    st.session_state.uid = uid
    st.session_state.chat = []          # [{"role", "content", "steps"?}]
    st.session_state.sent = 0


def run_turn(text: str) -> None:
    agent: SupportAgent = st.session_state.agent
    st.session_state.chat.append({"role": "user", "content": text})
    if st.session_state.sent >= config.DEMO_MAX_MESSAGES_PER_SESSION:
        st.session_state.chat.append({"role": "assistant", "content":
            "Bu demo oturumunun mesaj sınırına ulaşıldı. Kenar çubuğundan oturumu yenileyebilirsiniz."})
        return
    st.session_state.sent += 1
    with st.spinner("Asistan düşünüyor…"):
        turn = agent.run(text)
    steps = [{"tool": c.name, "args": _short_args(c.arguments), "result": result_code(c.result)}
             for c in turn.tool_calls]
    st.session_state.chat.append({"role": "assistant", "content": turn.answer, "steps": steps,
                                  "summary": turn.summary})


def _short_args(args: Optional[dict]) -> str:
    """Ara adımlarda sipariş numarasını kısaltarak göster (ekran görüntüsünde tam id görünmesin)."""
    if not isinstance(args, dict):
        return "(geçersiz JSON)"
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) == 32:
            v = v[:6] + "…"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def decide(approve: bool) -> None:
    agent: SupportAgent = st.session_state.agent
    p = agent.ctx.pending
    if p is None:
        return
    if approve:
        out = commit_pending(agent.ctx, p.proposal_id, approved_by=f"customer-ui:{hash_id(st.session_state.uid)}")
        note = (f"İade talebi oluşturulamadı: {out['message']}" if "error" in out else
                f"İade talebiniz oluşturuldu: {out['return_request_id']} · {out['status']} · "
                f"{out['refund_amount_brl']:.2f} BRL")
    else:
        reject_pending(agent.ctx)
        note = "İade talebi oluşturulmadı."
    agent.history.append({"role": "assistant", "content": note})
    st.session_state.chat.append({"role": "assistant", "content": note, "approval": approve})


# --------------------------------------------------------------------------- #
# Sayfa
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        db = get_db()
    except Exception as exc:  # bulutta ilk açılışta indirme/kurulum hatası: çökme yerine açıklama
        st.error(f"Veri hazırlanamadı ({type(exc).__name__}). Olist veri seti indirilemedi ya da DB "
                 "kurulamadı. Birkaç dakika sonra sayfayı yenileyin.")
        st.stop()
    scenarios = scenario_customers(db)

    with st.sidebar:
        st.header("Oturum")
        st.caption("Kimlik doğrulama simülasyonu: müşteri burada seçilir. Sohbette yazılan hiçbir şey "
                   "(\"ben yöneticiyim\", başka bir müşteri numarası) oturumu değiştirmez.")
        labels = [label for label, _ in scenarios]
        choice = st.selectbox("Demo müşterisi", labels, index=0)
        uid = dict(scenarios)[choice]
        refresh = st.button("Oturumu yenile", width="stretch")  # önce çiz: kısa devre gizlemesin
        if refresh or st.session_state.get("uid") != uid:
            start_session(uid, db)
        st.caption(f"Müşteri #{hash_id(uid)}")
        st.subheader("Siparişleriniz")
        for order in customer_orders(db, uid):
            st.caption(f"{order['Durum']} · {order['Sipariş tarihi']}")
            st.code(order["Sipariş no"], language=None)  # sağ üstte kopyala düğmesi
        st.caption("Sipariş numarasını kopyalayıp soruya yapıştırabilirsiniz.")
        st.divider()
        llm = st.session_state.get("llm_override") or get_llm()
        st.caption(f"Bugün (simüle): **{config.SIMULATED_TODAY}**  \n"
                   f"Model: `{getattr(llm, 'model', '—')}`  \n"
                   f"Politika arama: `{get_index().embedder.name}`")
        st.caption("Örnekpazar kurgusaldır. Veri: Olist Brazilian E-Commerce (CC BY-NC-SA 4.0).")

    st.title("🛍️ Örnekpazar Müşteri Destek")
    st.caption("Sipariş durumu, iade, iptal, kargo ve ödeme soruları. İade kararı ve tutarı kural "
               "motorundan gelir; talep yalnız sizin onayınızla oluşturulur.")

    if st.session_state.get("agent") is None:
        st.error("LLM anahtarı tanımlı değil. Yerelde `.env`, Streamlit Cloud'da Secrets içine "
                 "`GEMINI_API_KEY` ekleyin.")
        return

    for i, msg in enumerate(st.session_state.chat):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps") is not None:
                s = msg.get("summary", {})
                with st.expander(f"Ara adımlar · {len(msg['steps'])} tool · {s.get('steps', 0)} LLM adımı · "
                                 f"{s.get('prompt_tokens', 0) + s.get('completion_tokens', 0)} token · "
                                 f"{s.get('work_latency_s', 0)} sn"):
                    if not msg["steps"]:
                        st.caption("Tool çağrılmadı.")
                    for step in msg["steps"]:
                        st.markdown(f"- `{step['tool']}`({step['args']}) → **{step['result']}**")

    agent: SupportAgent = st.session_state.agent
    pending = agent.ctx.pending
    if pending is not None:
        with st.container(border=True):
            st.subheader("İade talebi onayı")
            c1, c2, c3 = st.columns(3)
            c1.metric("İade tutarı", f"{pending.refund_amount:.2f} BRL")
            c2.metric("Gerekçe", REASON_TR.get(pending.reason, pending.reason))
            c3.metric("Onay kuyruğu", "Kıdemli onay" if pending.approval_status == "PENDING_SENIOR"
                      else "Standart inceleme")
            detail = f"Sipariş {pending.order_id[:6]}… · ödeme yöntemine {pending.original_method_refund:.2f} BRL"
            if pending.voucher_refund:
                detail += f" · hediye çekine {pending.voucher_refund:.2f} BRL"
            if pending.freight_only:
                detail += " · yalnız kargo ücreti"
            st.caption(detail)
            b1, b2 = st.columns(2)
            b1.button("✅ Onayla ve talebi oluştur", type="primary", width="stretch",
                      on_click=decide, args=(True,))
            b2.button("Vazgeç", width="stretch", on_click=decide, args=(False,))

    if not st.session_state.chat:
        st.caption("Örnek sorular:")
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            col.button(ex, width="stretch", on_click=run_turn, args=(ex,))

    if prompt := st.chat_input("Sorunuzu yazın…", disabled=pending is not None):
        run_turn(prompt)
        st.rerun()
    if pending is not None:
        st.caption("Devam etmek için önce talebi onaylayın ya da vazgeçin.")


main()
