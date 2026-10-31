"""İade/iptal kural motoru (R0–R13). LLM'siz, deterministik, saf Python.

[Ref: 02_ShopOps_Eticaret.pdf §6.1]

İki katman:
- evaluate(): saf fonksiyon. Sipariş gerçeklerini (OrderFacts) ve gerekçeyi alır, kararı döner.
  DB, saat ya da LLM bilmez; bu yüzden birim testleri hızlı ve kesindir.
- check_return_eligibility(): DB'den gerçekleri oturumdaki customer_unique_id ile okur
  (R1 yetki burada, SQL JOIN'inde) ve evaluate()'i çağırır.

Tarihler takvim günü olarak karşılaştırılır; saat atılır; teslim günü 0. gündür. [§6]
"Bugün" her zaman parametredir; varsayılanı config.SIMULATED_TODAY. [§11]

Değerlendirme sırası (ilk karar veren adımda durulur; bayrak ve tutar sonra eklenir) [§6.1]:
  1. R0, R1 → NOT_FOUND
  2. R2–R5 durum kontrolleri
  3. R11 → DUPLICATE_REQUEST
  4. R6/R7 pencere → ELIGIBLE / NOT_ELIGIBLE_WINDOW
  5. R9 bayrağı   6. R8 tutarı   7. R10 onay seviyesi; R12, R13 bilgi bayrakları
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

import config


class Reason(str, Enum):
    CHANGED_MIND = "changed_mind"
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    LATE = "late"


class Decision(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    NOT_DELIVERED_CANCELLABLE = "NOT_DELIVERED_CANCELLABLE"
    IN_TRANSIT = "IN_TRANSIT"
    ALREADY_CLOSED = "ALREADY_CLOSED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    DUPLICATE_REQUEST = "DUPLICATE_REQUEST"
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE_WINDOW = "NOT_ELIGIBLE_WINDOW"


class ApprovalStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    PENDING_SENIOR = "PENDING_SENIOR"


# R1: sipariş hiç yokken ve başkasınınken aynı mesaj. Varlık ifşa edilmez.
NOT_FOUND_MESSAGE = "Bu numarayla size ait bir sipariş bulunamadı."

CANCELLABLE_STATUSES = frozenset({"created", "approved", "invoiced", "processing"})   # R2
CLOSED_STATUSES = frozenset({"canceled", "unavailable"})                              # R4
# R11: bu statülerden birinde talep varsa ikincisi açılmaz (tabloda başka statü yok).
BLOCKING_REQUEST_STATUSES = frozenset({"PENDING_REVIEW", "PENDING_SENIOR", "APPROVED", "REJECTED"})

STANDARD_WINDOW_REASONS = frozenset({Reason.CHANGED_MIND, Reason.LATE})   # R6
EXTENDED_WINDOW_REASONS = frozenset({Reason.DAMAGED, Reason.WRONG_ITEM})  # R7


def to_day(value: Optional[str]) -> Optional[date]:
    """'2018-08-18 23:59:00' → date(2018, 8, 18). Boş/NULL → None. Saat atılır."""
    if value is None or str(value).strip() == "":
        return None
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class Item:
    price: float
    freight_value: float
    shipping_limit_date: Optional[date]


@dataclass(frozen=True)
class OrderFacts:
    """Kural motorunun ihtiyaç duyduğu, tek bir siparişe ait gerçekler."""

    order_id: str
    status: str
    purchase_date: Optional[date]
    delivered_date: Optional[date]
    estimated_date: Optional[date]
    carrier_date: Optional[date]
    items: tuple[Item, ...] = ()
    payment_total: float = 0.0
    voucher_total: float = 0.0
    existing_request_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class EligibilityResult:
    """check_return_eligibility çıktısı. Tool katmanı bunu JSON'a çevirir."""

    decision: Decision
    rule: str                                   # kararı veren kural, ör. "R6"
    reason: Optional[Reason] = None
    refund_amount: float = 0.0                  # R8
    freight_only: bool = False                  # R9: pencere dışı, yalnız kargo talebi
    freight_refund_flag: bool = False           # R9
    seller_caused_delay: bool = False           # R13
    approval_status: Optional[ApprovalStatus] = None  # R10; talep açılamıyorsa None
    voucher_refund: float = 0.0                 # R12
    original_method_refund: float = 0.0         # R12
    days_since_delivery: Optional[int] = None
    late_days: Optional[int] = None             # teslim − tahmini teslim (takvim günü)
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def can_open_request(self) -> bool:
        """create_return_request çağrılabilir mi? (ELIGIBLE ya da R9 kargo talebi)"""
        return self.approval_status is not None and self.refund_amount > 0

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "rule": self.rule,
            "reason": self.reason.value if self.reason else None,
            "refund_amount": self.refund_amount,
            "freight_only": self.freight_only,
            "freight_refund_flag": self.freight_refund_flag,
            "seller_caused_delay": self.seller_caused_delay,
            "approval_status": self.approval_status.value if self.approval_status else None,
            "voucher_refund": self.voucher_refund,
            "original_method_refund": self.original_method_refund,
            "days_since_delivery": self.days_since_delivery,
            "late_days": self.late_days,
            "can_open_request": self.can_open_request,
        }


def parse_reason(value: str | Reason) -> Reason:
    """Geçersiz gerekçe tahmin edilmez; ValueError ile reddedilir."""
    try:
        return value if isinstance(value, Reason) else Reason(str(value).strip().lower())
    except ValueError:
        allowed = ", ".join(r.value for r in Reason)
        raise ValueError(f"Geçersiz iade gerekçesi: {value!r}. İzinli: {allowed}") from None


def _money(x: float) -> float:
    return round(x + 1e-9, 2)


def _seller_caused(facts: OrderFacts, today: date) -> bool:
    """R13: kargoya veriliş günü herhangi bir kalemin son kargolama gününden sonra mı?

    Yalnız bugüne kadar gerçekleşmiş kargo tarihleri dikkate alınır.
    """
    if facts.carrier_date is None or facts.carrier_date > today:
        return False
    return any(i.shipping_limit_date is not None and facts.carrier_date > i.shipping_limit_date
               for i in facts.items)


def evaluate(facts: Optional[OrderFacts], reason: str | Reason,
             today: date = config.SIMULATED_TODAY) -> EligibilityResult:
    """R0–R13'ü belgedeki sırayla uygular. facts=None → NOT_FOUND (R1)."""
    reason = parse_reason(reason)

    # 1. R0, R1 -----------------------------------------------------------------
    if facts is None:
        return EligibilityResult(Decision.NOT_FOUND, "R1", reason)
    if facts.purchase_date is None or facts.purchase_date >= today:
        return EligibilityResult(Decision.NOT_FOUND, "R0", reason)  # yükleyiciye ek savunma

    # 2. R2–R5 -----------------------------------------------------------------
    status = facts.status
    if status in CANCELLABLE_STATUSES:
        return EligibilityResult(Decision.NOT_DELIVERED_CANCELLABLE, "R2", reason)
    if status == "shipped" or (status == "delivered" and facts.delivered_date is not None
                               and facts.delivered_date > today):
        return EligibilityResult(Decision.IN_TRANSIT, "R3", reason)
    if status in CLOSED_STATUSES:
        return EligibilityResult(Decision.ALREADY_CLOSED, "R4", reason)
    if status != "delivered" or facts.delivered_date is None:
        # R5; tanımsız bir statü de insana devredilir (tahmin yok).
        return EligibilityResult(Decision.NEEDS_HUMAN, "R5", reason)

    # Buradan sonra: delivered ve teslim ≤ bugün.
    days_since = (today - facts.delivered_date).days
    late_days = ((facts.delivered_date - facts.estimated_date).days
                 if facts.estimated_date is not None else None)
    r9 = late_days is not None and late_days >= config.FREIGHT_REFUND_LATE_DAYS
    r13 = _seller_caused(facts, today)
    common = dict(reason=reason, days_since_delivery=days_since, late_days=late_days,
                  freight_refund_flag=r9, seller_caused_delay=r13)

    # 3. R11 -------------------------------------------------------------------
    if any(s in BLOCKING_REQUEST_STATUSES for s in facts.existing_request_statuses):
        return EligibilityResult(Decision.DUPLICATE_REQUEST, "R11", **common)

    # 4. R6 / R7 ---------------------------------------------------------------
    if reason in EXTENDED_WINDOW_REASONS:
        window, rule = config.RETURN_WINDOW_DAYS_EXTENDED, "R7"
    else:
        window, rule = config.RETURN_WINDOW_DAYS_STANDARD, "R6"
    eligible = 0 <= days_since <= window

    # 5–6. R9 bayrağı ve R8 tutarı -----------------------------------------------
    price_total = sum(i.price for i in facts.items)
    freight_total = sum(i.freight_value for i in facts.items)
    freight_only = False
    if eligible:
        include_freight = rule == "R7" or r9
        refund = price_total + (freight_total if include_freight else 0.0)
    elif r9 and days_since <= config.RETURN_WINDOW_DAYS_EXTENDED:
        refund, freight_only = freight_total, True
    else:
        refund = 0.0
    refund = _money(refund)

    # 7. R10 onay seviyesi; R12 ödeme dağılımı ------------------------------------
    approval = None
    if refund > 0:
        approval = (ApprovalStatus.PENDING_SENIOR
                    if refund > config.SENIOR_APPROVAL_THRESHOLD_BRL
                    else ApprovalStatus.PENDING_REVIEW)
    voucher_share = (facts.voucher_total / facts.payment_total) if facts.payment_total > 0 else 0.0
    voucher_refund = _money(refund * voucher_share)

    flags = tuple(f for f, on in (("R9_FREIGHT_REFUND", r9), ("R13_SELLER_DELAY", r13)) if on)
    return EligibilityResult(
        Decision.ELIGIBLE if eligible else Decision.NOT_ELIGIBLE_WINDOW, rule,
        refund_amount=refund, freight_only=freight_only, approval_status=approval,
        voucher_refund=voucher_refund, original_method_refund=_money(refund - voucher_refund),
        flags=flags, **common,
    )


# --------------------------------------------------------------------------- #
# DB katmanı
# --------------------------------------------------------------------------- #
_ORDER_FOR_CUSTOMER_SQL = """
SELECT o.order_id, o.order_status, o.order_purchase_timestamp,
       o.order_delivered_customer_date, o.order_estimated_delivery_date,
       o.order_delivered_carrier_date
FROM orders o JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_id = ? AND c.customer_unique_id = ?
"""


def load_order_facts(con: sqlite3.Connection, order_id: str,
                     customer_unique_id: str) -> Optional[OrderFacts]:
    """Siparişi yalnız oturumdaki müşteriye aitse döner (R1). Değilse None.

    Yetki filtresi SQL'in içindedir: başkasının siparişi hiç okunmaz.
    """
    if not order_id or not customer_unique_id:
        return None
    row = con.execute(_ORDER_FOR_CUSTOMER_SQL, (order_id.strip(), customer_unique_id)).fetchone()
    if row is None:
        return None
    oid = row[0]
    items = tuple(
        Item(float(p or 0), float(f or 0), to_day(lim)) for p, f, lim in con.execute(
            "SELECT price, freight_value, shipping_limit_date FROM order_items WHERE order_id = ?",
            (oid,))
    )
    pay_total, voucher_total = con.execute(
        "SELECT COALESCE(SUM(payment_value), 0), "
        "COALESCE(SUM(CASE WHEN payment_type = 'voucher' THEN payment_value END), 0) "
        "FROM order_payments WHERE order_id = ?", (oid,)).fetchone()
    requests = tuple(r[0] for r in con.execute(
        "SELECT status FROM return_requests WHERE order_id = ?", (oid,)))
    return OrderFacts(
        order_id=oid, status=row[1], purchase_date=to_day(row[2]), delivered_date=to_day(row[3]),
        estimated_date=to_day(row[4]), carrier_date=to_day(row[5]), items=items,
        payment_total=float(pay_total), voucher_total=float(voucher_total),
        existing_request_statuses=requests,
    )


def check_return_eligibility(con: sqlite3.Connection, order_id: str, customer_unique_id: str,
                             reason: str | Reason,
                             today: date = config.SIMULATED_TODAY) -> EligibilityResult:
    """Tool katmanının çağıracağı giriş noktası. Kimlik parametre olarak LLM'den gelmez;
    çağıran katman oturumdan verir. [Ref: §5.3 "Kimlik ilkesi"]"""
    return evaluate(load_order_facts(con, order_id, customer_unique_id), reason, today)

