from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    hours_24h: int = 24
    days_7d: int = 7


def get_recent_messages(
    messages: pd.DataFrame,
    user_id: str,
    before_ts: pd.Timestamp,
    hours: int,
) -> pd.DataFrame:
    """
    Return messages for a user in the window (before_ts - hours, before_ts].
    """
    m = messages[messages["user_id"] == user_id]
    start = before_ts - pd.Timedelta(hours=hours)
    return m[(m["timestamp"] > start) & (m["timestamp"] <= before_ts)].sort_values("timestamp")


def message_summary(messages_window: pd.DataFrame) -> dict[str, Any]:
    if messages_window.empty:
        return {
            "count": 0,
            "phish_like_count": 0,
            "top_senders": [],
            "keyword_hits": 0,
        }

    text = messages_window.get("message_text", pd.Series([], dtype=str)).fillna("").astype(str).str.lower()
    keywords = ["verify", "urgent", "locked", "confirm", "fee", "pay", "password", "code", "voucher", "re-authenticate"]
    keyword_hits = int(sum(text.str.contains(k, regex=False).sum() for k in keywords))

    phish_like = 0
    if "is_phishing_pred" in messages_window.columns:
        phish_like = int(messages_window["is_phishing_pred"].astype(int).sum())
    elif "is_phishing" in messages_window.columns:
        phish_like = int(messages_window["is_phishing"].astype(int).sum())

    top_senders = (
        messages_window.get("sender", pd.Series([], dtype=str))
        .fillna("")
        .astype(str)
        .value_counts()
        .head(3)
        .index.tolist()
    )

    return {
        "count": int(messages_window.shape[0]),
        "phish_like_count": int(phish_like),
        "top_senders": top_senders,
        "keyword_hits": int(keyword_hits),
    }


def transaction_risk_signals(tx: pd.Series) -> list[str]:
    """
    Human-friendly risk signals computed from a single transaction row.
    These are used both for explanations and as inputs to an LLM.
    """
    signals: list[str] = []

    if str(tx.get("entry_mode", "")).lower() == "online":
        signals.append("online card entry")
    if int(tx.get("device_seen_before", 1)) == 0:
        signals.append("new / unseen device")
    if int(tx.get("merchant_seen_before", 1)) == 0:
        signals.append("new merchant")
    if int(tx.get("is_international", 0)) == 1:
        signals.append("international merchant")
    if str(tx.get("merchant_category", "")).lower() in {"gift_cards", "crypto"}:
        signals.append(f"sensitive category: {tx.get('merchant_category')}")
    if float(tx.get("amount_eur", 0.0)) >= 600:
        signals.append("high amount")
    if int(tx.get("is_weekend", 0)) == 1 and int(tx.get("hour", 12)) <= 6:
        signals.append("off-hours weekend activity")

    return signals


def format_transaction_context(tx: pd.Series, msg_summary: dict[str, Any], signals: list[str]) -> str:
    """
    Compact textual context passed to an LLM.
    """
    parts = []
    parts.append("Transaction:")
    parts.append(f"- transaction_id: {tx.get('transaction_id')}")
    parts.append(f"- user_id: {tx.get('user_id')}")
    parts.append(f"- amount_eur: {tx.get('amount_eur')}")
    parts.append(f"- merchant: {tx.get('merchant_name')} ({tx.get('merchant_category')})")
    parts.append(f"- entry_mode: {tx.get('entry_mode')} / channel: {tx.get('channel')}")
    parts.append(f"- merchant_country: {tx.get('merchant_country')} / ip_country: {tx.get('ip_country')}")
    parts.append(f"- device_seen_before: {tx.get('device_seen_before')} / merchant_seen_before: {tx.get('merchant_seen_before')}")
    parts.append("")
    parts.append("Recent messages (window):")
    parts.append(f"- count: {msg_summary.get('count')}")
    parts.append(f"- phish_like_count: {msg_summary.get('phish_like_count')}")
    parts.append(f"- keyword_hits: {msg_summary.get('keyword_hits')}")
    parts.append(f"- top_senders: {msg_summary.get('top_senders')}")
    parts.append("")
    parts.append("Computed risk signals:")
    parts.append("- " + "; ".join(signals) if signals else "- none")
    return "\n".join(parts)

