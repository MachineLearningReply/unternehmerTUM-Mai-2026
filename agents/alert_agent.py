from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .llm import LLMClient, LLMError
from .tools import format_transaction_context, get_recent_messages, message_summary, transaction_risk_signals


@dataclass(frozen=True)
class AlertDecision:
    is_suspicious: int
    severity: str
    recommended_action: str
    explanation: str
    rationale_bullets: list[str]
    risk_factors: list[str]


SYSTEM_PROMPT = """You are a bank fraud analyst assistant.
You receive a single transaction and a summary of recent messages.
Return ONLY valid JSON with these keys:
{
  "is_suspicious": 0|1,
  "severity": "low"|"medium"|"high",
  "recommended_action": "ignore"|"review"|"step_up_auth"|"block",
  "explanation": "short plain-language sentence",
  "rationale_bullets": ["...", "..."],
  "risk_factors": ["...", "..."]
}

Be conservative: if evidence is weak, prefer "review" over "block".
"""


class AlertAgent:
    def __init__(self, llm: LLMClient, messages: pd.DataFrame, message_window_hours: int = 24):
        self._llm = llm
        self._messages = messages
        self._window_h = int(message_window_hours)

    def decide(self, tx_row: pd.Series) -> AlertDecision:
        before_ts = pd.to_datetime(tx_row["timestamp"], utc=True)
        recent = get_recent_messages(self._messages, user_id=str(tx_row["user_id"]), before_ts=before_ts, hours=self._window_h)
        summary = message_summary(recent)
        signals = transaction_risk_signals(tx_row)
        context = format_transaction_context(tx_row, msg_summary=summary, signals=signals)

        try:
            obj = self._llm.complete_json(system=SYSTEM_PROMPT, user=context)
        except LLMError:
            # Hard fallback: deterministic explanation
            return AlertDecision(
                is_suspicious=1 if signals else 0,
                severity="medium" if signals else "low",
                recommended_action="review" if signals else "ignore",
                explanation="Flagged due to unusual context and risk signals." if signals else "No strong risk signals detected.",
                rationale_bullets=signals[:4] if signals else ["none"],
                risk_factors=[s.replace(" ", "_") for s in signals[:6]],
            )

        return AlertDecision(
            is_suspicious=int(obj.get("is_suspicious", 0)),
            severity=str(obj.get("severity", "low")),
            recommended_action=str(obj.get("recommended_action", "review")),
            explanation=str(obj.get("explanation", "")),
            rationale_bullets=[str(x) for x in (obj.get("rationale_bullets") or [])][:8],
            risk_factors=[str(x) for x in (obj.get("risk_factors") or [])][:12],
        )

