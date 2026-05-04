from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMClient:
    """
    Minimal interface used by the agent components.

    The workshop pack ships with a `MockLLMClient` so everything runs offline.
    Teams can plug in a real LLM by using `OpenAICompatibleClient` or by writing
    their own adapter.
    """

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """
        Best-effort JSON extraction: expects the model to return JSON, but will try
        to salvage a JSON object if it's wrapped in text.
        """
        text = self.complete(system=system, user=user)
        return extract_json_object(text)


def extract_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise LLMError("Empty LLM response")

    s = text.strip()
    # Direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find a JSON object in the text (common for chatty outputs).
    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise LLMError("Could not parse JSON from LLM response")


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """
    Generic config for an OpenAI-compatible Chat Completions endpoint.

    This is intentionally provider-agnostic. If your provider uses a different
    schema, adapt `OpenAICompatibleClient.complete`.
    """

    base_url: str
    api_key: str
    model: str = "gpt-4o-mini"
    timeout_s: int = 60


class OpenAICompatibleClient(LLMClient):
    def __init__(self, cfg: OpenAICompatibleConfig):
        self._cfg = cfg

    def complete(self, system: str, user: str) -> str:
        url = self._cfg.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._cfg.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as e:  # pragma: no cover
            raise LLMError(f"LLM request failed: {e}") from e

        try:
            obj = json.loads(raw)
            return obj["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Unexpected LLM response format: {e}") from e


class MockLLMClient(LLMClient):
    """
    Offline mock that returns deterministic JSON for demos/tests.
    This is NOT a real LLM; it exists to make the agentic pipeline runnable
    without internet/API keys.
    """

    def complete(self, system: str, user: str) -> str:
        if "\"ideas\"" in (system or "") or "Propose" in (user or "") or "improvement ideas" in (user or "").lower():
            obj = {
                "ideas": [
                    {
                        "name": "msg_intent_feature",
                        "type": "feature",
                        "description": "Classify messages into intents (OTP, delivery fee, invoice update) and aggregate as priors.",
                        "how_to_implement": "Run a text classifier/LLM over messages → intent labels; for each txn compute intent counts in last 24h/7d.",
                        "why_it_helps": "Social-engineering and takeover often precede suspicious transfers/gift cards.",
                        "risks": "Heuristics may mislabel marketing; validate on train only to avoid leakage.",
                    },
                    {
                        "name": "velocity_burst",
                        "type": "feature",
                        "description": "Add burst/velocity features for short windows (10m/1h).",
                        "how_to_implement": "Per user, compute rolling counts and rolling sum(amount) before each txn.",
                        "why_it_helps": "Card testing and cash-out patterns show clustered activity.",
                        "risks": "Legitimate travel spikes; add travel context and cap alerting via budgets.",
                    },
                    {
                        "name": "new_beneficiary_rule",
                        "type": "rule",
                        "description": "Flag large transfers to a new beneficiary combined with new device/IP change.",
                        "how_to_implement": "Track beneficiary_seen_before per user; if amount>800 and new beneficiary and device_seen_before==0 then flag.",
                        "why_it_helps": "Targets account takeover cash-out transfers.",
                        "risks": "Some legitimate first-time transfers; use step-up auth instead of block.",
                    },
                    {
                        "name": "drift_monitor_crypto",
                        "type": "monitor",
                        "description": "Monitor share of crypto-category transactions and alert on sudden spikes (drift).",
                        "how_to_implement": "Daily/weekly metric: % merchant_category==crypto; compare to rolling baseline; trigger investigation agent.",
                        "why_it_helps": "Test split contains more crypto-ramp fraud; this catches distribution shifts.",
                        "risks": "Crypto adoption growth may be legitimate; monitor + review, don't auto-block.",
                    },
                    {
                        "name": "calibrate_threshold_by_budget",
                        "type": "rule",
                        "description": "Pick threshold by a review budget (top-K per day) rather than a fixed probability.",
                        "how_to_implement": "Sort transactions by risk_score per day; flag top K; evaluate cost trade-offs.",
                        "why_it_helps": "Matches real operational constraints and prevents alert floods.",
                        "risks": "May miss low-score fraud on busy days; combine with hard rules for extreme cases.",
                    },
                ]
            }
            return json.dumps(obj, ensure_ascii=False)

        text = (user or "").lower()
        risk_factors: list[str] = []
        if "online" in text:
            risk_factors.append("online_transaction")
        if "new / unseen device" in text or "unseen device" in text or "device_seen_before: 0" in text:
            risk_factors.append("new_device")
        if "international" in text or "is_international: 1" in text:
            risk_factors.append("international")
        if "phishing" in text or "phish" in text:
            risk_factors.append("recent_phishing_signal")
        if "gift" in text and "card" in text:
            risk_factors.append("gift_cards")
        if "crypto" in text:
            risk_factors.append("crypto_ramp")

        severity = "low"
        if any(x in risk_factors for x in ["recent_phishing_signal", "new_device"]) or "amount_eur" in text:
            severity = "medium"
        if any(x in risk_factors for x in ["crypto_ramp", "gift_cards"]) and "phish" in text:
            severity = "high"

        recommended_action = "review"
        if severity == "high":
            recommended_action = "step_up_auth"

        obj = {
            "is_suspicious": 1 if severity in {"medium", "high"} else 0,
            "severity": severity,
            "recommended_action": recommended_action,
            "explanation": "Flagged due to a combination of unusual context and risk signals.",
            "rationale_bullets": [f.replace("_", " ") for f in risk_factors[:5]] or ["unusual pattern vs history"],
            "risk_factors": risk_factors,
        }
        return json.dumps(obj, ensure_ascii=False)


def llm_from_env() -> LLMClient:
    """
    Create an LLM client from environment variables.

    - Default: mock (offline)
    - Set LLM_MODE=openai_compatible and provide:
        - LLM_BASE_URL
        - LLM_API_KEY
        - LLM_MODEL (optional)
    """
    mode = (os.getenv("LLM_MODE") or "mock").strip().lower()
    if mode == "mock":
        return MockLLMClient()

    if mode == "openai_compatible":
        base_url = os.getenv("LLM_BASE_URL") or ""
        api_key = os.getenv("LLM_API_KEY") or ""
        model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
        if not base_url or not api_key:
            raise LLMError("LLM_MODE=openai_compatible requires LLM_BASE_URL and LLM_API_KEY")
        return OpenAICompatibleClient(OpenAICompatibleConfig(base_url=base_url, api_key=api_key, model=model))

    raise LLMError(f"Unknown LLM_MODE: {mode}")
