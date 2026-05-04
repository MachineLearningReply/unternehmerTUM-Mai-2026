# Challenge Brief — Adaptive AI Agents for Fraud Detection

## Context

You are a team building a prototype fraud detection system for a retail bank. Your system should identify suspicious transactions under:

- **Uncertainty and changing behavior** (new merchants, new fraud tactics, customer travel)
- **Unequal costs of error** (missing fraud is usually more expensive than reviewing a false alarm)
- **Operational constraints** (analyst review capacity; need for explanations)

You will work with a synthetic dataset containing:

- `users.csv`: customer profiles
- `transactions.csv`: transaction stream (train labeled)
- `messages.csv`: SMS/email messages received by customers (train labeled as phishing)

In the test split, labels are hidden in the public files and concept drift is present (a new-ish fraud pattern appears more frequently).

## Your task

Build a system that:

1) Scores transactions by fraud risk (0–1 or 0–100)
2) Produces a list of **flagged `transaction_id`s** as “fraud suspects”
3) Provides a short **explanation** for why each transaction was flagged
4) Chooses an operating point (threshold) based on an explicit **cost function**

## Deliverables (what you submit)

### A) Submission file (required)

Submit **one file** containing the transaction ids you flag as fraud.

Accepted formats:
- CSV with a column named `transaction_id`
- Plain text with one `transaction_id` per line

Example CSV:

```csv
transaction_id
T0000123
T0000456
```

### B) Short write-up / README (required)

1–2 pages (or a short `README.md`) covering:
- Your approach (model/rules + features used)
- Your cost assumptions (review cost vs missed fraud cost)
- How you used messages/text (even simple heuristics count)
- Limitations + next improvements (drift, monitoring, human review)

### C) Presentation (required)

3–5 slides, 5 minutes:
- Problem framing + operational goal
- 2–3 EDA insights
- Model/rule and evaluation results
- 2–3 example alerts with explanations
- Next steps (deployment, monitoring, governance)

### D) Code (required)

Reproducible code in a notebook or script:
- Data loading
- Feature engineering
- Training / scoring
- Submission generation

### E) Agentic/LLM component (required)

Include at least one “agentic” element that uses an LLM in a controlled way, for example:
- an **Alert Agent** that generates structured explanations + recommended actions
- an LLM-assisted **message understanding** step (intent/phishing cues → features)
- an “investigation agent” that proposes new rules/features when performance drops on drift

In your write-up, document:
- the prompt(s) you used (or the function/tool schema)
- how you ensured structured output (JSON)
- fallback behavior when the LLM fails (timeouts, invalid JSON)

## Suggested approach (baseline → better)

**Baseline ideas**
- Flag “new device + online + high amount”
- Use message signals: “phishing message in last 24h”

**Model ideas**
- Logistic regression or tree-based model on engineered features
- Text features from messages:
  - keyword counts
  - TF‑IDF
  - LLM-assisted classification (phish intent / urgency / OTP / payment request)

**Agent idea (optional)**
- “Alert agent” that generates analyst/customer-friendly explanations from the top contributing signals.

## Scoring (combined)

Total = **100 points**.

### 1) Offline detection score (60 points)

Computed by the organizer’s evaluation script:
- **Cost Score (0–100)** based on missed fraud loss and review/false-positive cost
- **F2 score (0–1)** to emphasize recall (catching fraud) while still penalizing low precision

The organizer uses a weighted combination (default):
- 70% Cost Score
- 30% F2 score

Mapped to 0–60 points.

### 2) Explainability & decision workflow (15 points)

- Clear reasons for flags (features/signals) (6)
- Clear operational action (review vs block vs step-up auth) (5)
- Agentic/LLM implementation quality (JSON structure, robustness, fallbacks) (4)

### 3) Business/risk framing (15 points)

- Explicit asymmetric costs and threshold rationale (10)
- Reasonable review budget / trade-offs (5)

### 4) Presentation quality (10 points)

- Clear story + visuals + honest limitations (10)

## Rules / constraints

- Do not use test labels (you should only receive the public test files).
- You may use LLMs for feature ideas, code generation, and text understanding, but your output must be reproducible (save prompts or describe them).
- Keep solutions lightweight enough to run locally within the workshop.
