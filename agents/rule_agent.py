from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .llm import LLMClient, LLMError


SYSTEM_PROMPT = """You are a fraud risk engineer helping a bank improve a detection system.
You will receive summary statistics and a small sample of error cases.
Return ONLY valid JSON with:
{
  "ideas": [
    {
      "name": "short_name",
      "type": "feature|rule|monitor",
      "description": "1 sentence",
      "how_to_implement": "practical steps in Python",
      "why_it_helps": "how it targets the observed errors",
      "risks": "possible false positives / leakage / fairness concerns"
    }
  ]
}

Focus on:
- concept drift signals
- behavior/velocity features
- using message intent as a prior
- operational thresholds and review budgets
"""


@dataclass(frozen=True)
class RuleIdeas:
    ideas: list[dict[str, Any]]


class RuleProposalAgent:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def propose(self, error_cases: pd.DataFrame, notes: str = "") -> RuleIdeas:
        sample_df = error_cases.head(12).copy()
        # Make JSON-serializable (timestamps, numpy types, etc.)
        for col in sample_df.columns:
            if "time" in col.lower() or str(sample_df[col].dtype).startswith("datetime"):
                sample_df[col] = sample_df[col].astype(str)
        sample = sample_df.to_dict(orient="records")
        user = "Error cases sample (JSON):\n" + json.dumps(sample, ensure_ascii=False) + "\n\n"
        if notes:
            user += "Notes:\n" + notes.strip() + "\n\n"
        user += "Propose 6-10 improvement ideas."

        try:
            obj = self._llm.complete_json(system=SYSTEM_PROMPT, user=user)
        except LLMError:
            return RuleIdeas(
                ideas=[
                    {
                        "name": "recent_phish_prior",
                        "type": "feature",
                        "description": "Add a prior feature for phishing-like messages in the last 24h.",
                        "how_to_implement": "For each transaction, count messages in last 24h with URL/urgent keywords; merge as feature.",
                        "why_it_helps": "Many social-engineering and takeover cases are preceded by phishing messages.",
                        "risks": "May increase false positives for marketing messages; ensure no test-label leakage.",
                    },
                    {
                        "name": "velocity_features",
                        "type": "feature",
                        "description": "Add transaction velocity features (count/sum in last 10m/1h/24h).",
                        "how_to_implement": "Per user, rolling windows over timestamp to compute recent counts and total spend.",
                        "why_it_helps": "Card testing and cash-out often show bursts in short windows.",
                        "risks": "Travel spikes can look similar; include travel context signals.",
                    },
                ]
            )

        ideas = obj.get("ideas")
        if not isinstance(ideas, list):
            ideas = []
        return RuleIdeas(ideas=[x for x in ideas if isinstance(x, dict)][:12])
