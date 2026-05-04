# Adaptive AI Agents for Fraud Detection — Workshop Pack

This repo contains a synthetic, multi-modal dataset (tabular + transactions + messages) and a starter notebook for a 2.5–3 hour workshop where mixed teams (tech + business) build and present a fraud detection prototype.

## What students build (goal)

A small “fraud detection system” that:

- Flags suspicious transactions (binary label + risk score).
- Uses both **transaction behavior** and **text messages** (SMS/email) as signals.
- Produces **actionable explanations** (why flagged, what to do next).
- Optimizes for **unequal costs of error** (false negatives are usually more expensive than false positives).

## Suggested schedule (2.5–3 hours)

**Before (0–15 min): briefing**
- Problem framing: fraud types, asymmetry of costs, concept drift.
- Data overview + constraints.
- Team roles: risk/product (business), feature/modeling (tech), evaluation/storytelling (both).

**Build (15–140 min): workshop**
- 15–35: EDA + hypothesis generation (what looks suspicious?).
- 35–85: Baseline model (structured features only).
- 85–120: Add message/text features (keyword, TF‑IDF, or LLM-assisted extraction).
- 120–140: Thresholding with a cost function + explanation output.

**Present (140–180 min): demos + discussion**
- 5 min per team (3–4 slides + 1–2 minute “demo” of flagged alerts).
- Discussion: drift, calibration, human-in-the-loop, governance.

### Facilitator checkpoints (use as “during” slides)

- ~45 min: Each team shows 2 EDA insights + a first baseline (even if weak).
- ~90 min: Each team adds at least 1 message/text feature and explains why it should help.
- ~120 min: Each team picks a threshold based on a cost function (not accuracy).
- ~150 min: Each team prepares 2–3 concrete alert examples for the demo.

## Dataset

Generated files:
- `data/train/users.csv`
- `data/train/transactions.csv`
- `data/train/messages.csv`
- `data/test/users.csv`
- `data/test/transactions.csv` (includes labels)
- `data/test/messages.csv` (includes labels)
- `data/test/transactions_public.csv` (same as test transactions, but **without** fraud labels)
- `data/test/transactions_labels.csv` (label-only file for facilitators)
- `data/test/messages_public.csv` (messages without phishing labels)
- `data/test/messages_labels.csv` (label-only file for facilitators)
- `data/DATA_DICTIONARY.md` (column definitions)

The test split includes a small **concept drift**: new fraud pattern(s) that appear less often (or not at all) in training.

## Starter notebook

Notebook:
- `notebooks/Fraud_Detection_Workshop.ipynb`

It walks through:
- EDA (fraud prevalence, segments, channels, time-of-day)
- Feature engineering (user behavior + “recent phishing messages” features)
- Baseline model + evaluation (precision/recall, PR-AUC, cost-based thresholding)
- A simple “agent” style explanation function for alerts
- Optional hook points where teams can plug in an LLM for text understanding / rule generation

## How to run (local)

1) Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Start Jupyter:

```bash
jupyter lab
```

## Regenerate the dataset (optional)

The data is synthetic and reproducible:

```bash
python3 scripts/generate_datasets.py --out data --seed 42
```

## Challenge documents (handout + organizer)

- Student handout / scoring: `docs/CHALLENGE_BRIEF.md`
- Organizer troubleshooting guide: `docs/ORGANIZER_GUIDE.md`
- Evaluator (organizer): `eval/evaluate_submission.py`
- Baseline reference solution: `solutions/sample_solution.py`
- Agentic/LLM reference solution: `solutions/agentic_solution.py`
- Agentic building blocks (LLM stub + AlertAgent): `agents/`
- Investigation agent (rule/feature ideas): `solutions/rule_suggester.py`

## Student tasks (what to implement)

Minimum viable (everyone):
- Build a classifier or scoring rule for `transactions.is_fraud`.
- Choose a decision threshold that minimizes an explicit cost function.
- Produce a short explanation for each flagged transaction.

Good improvements (mixed teams):
- Create features that use `messages.csv` (e.g., “phishing in last 24h”, LLM-classified intent).
- Add behavior features (velocity, new device/merchant, unusual country, unusual amount vs. user baseline).
- Compare at least 2 approaches (e.g., logistic regression vs. tree model vs. rules).

Stretch goals:
- Concept drift monitoring: detect category shifts and performance degradation.
- Active learning: propose which transactions should be manually reviewed first.
- Fairness/robustness: analyze false positives across segments / employment statuses.

## What to present (template)

Slide 1 — Problem & approach:
- Which fraud types you target and why.
- What signals you used (transactions, messages, behavior).

Slide 2 — Data insights:
- 2–3 key EDA findings (e.g., fraud concentrated in online entry mode / new devices).

Slide 3 — Model & evaluation:
- Metrics (PR-AUC and precision/recall at your chosen threshold).
- Cost-based result (expected savings vs. baseline).

Slide 4 — Demo & next steps:
- 2–3 example alerts with explanations.
- How you would deploy/monitor (drift, feedback loop, human review).

## Evaluation rubric (simple)

- 40% Detection quality (recall at acceptable precision; cost-based objective)
- 25% Use of multi-modal signals (transactions + messages)
- 20% Explainability & decision workflow (what happens after an alert?)
- 15% Presentation clarity (story + limitations + next steps)

## Notes for facilitators (talking points)

Before:
- Emphasize asymmetric costs and why accuracy alone is insufficient.
- Encourage teams to decide an operational “review budget” (how many alerts/day).

During:
- Prompt business students to define: “What is the cost of a missed fraud vs. a false alarm?”
- Prompt technical students to show: “Which features move precision/recall the most?”

After:
- Discuss real-world constraints: privacy, compliance, model risk management, adversaries, feedback loops.
