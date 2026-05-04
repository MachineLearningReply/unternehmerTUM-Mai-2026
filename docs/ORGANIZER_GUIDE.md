# Organizer Guide — Fraud Detection Workshop

This guide is for the person running the challenge. It provides a structure for the session, “unstuck” hints, and how to evaluate submissions.

## What to share with students

Share:
- `/data/train/users.csv`
- `/data/train/transactions.csv`
- `/data/train/messages.csv`
- `/data/test/users.csv`
- `/data/test/transactions_public.csv`
- `/data/test/messages_public.csv`
- `/data/DATA_DICTIONARY.md`
- `/notebooks/Fraud_Detection_Workshop.ipynb` (starter notebook)
- `/docs/CHALLENGE_BRIEF.md`

Do **not** share:
- `/data/test/transactions_labels.csv`
- `/data/test/messages_labels.csv`
- `/data/test/transactions.csv` (contains labels)
- `/data/test/messages.csv` (contains labels)

Tip: put student files into a separate zip/folder (e.g., `student_pack/`) before distributing.

## Environment setup (quick)

If teams struggle with setup, suggest:

```bash
cd /Users/s.zanwar/Work/Reply/NLP/TUMunternehem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

If Jupyter is already available globally, they can skip venv and run `jupyter lab` directly.

## Suggested timeline (2.5–3 hours)

0–15 min: briefing
- Explain asymmetric costs + concept drift
- Show data schema (users / transactions / messages)
- Explain submission format: list of `transaction_id`s

15–45 min: EDA + baseline hypothesis
- Fraud rate
- Which channels / entry modes are risky?
- New device/merchant patterns

45–90 min: baseline model
- Structured features + a simple classifier
- Evaluate on a time-based validation split

90–120 min: add message features
- “phishing in last 24h” prior to transaction
- keyword/intent extraction

120–150 min: thresholding + explanations + packaging
- Pick a threshold using a cost function
- Produce 2–3 alert examples for the demo

150–180 min: presentations + discussion

## Common stuck points + quick hints

### “We don’t know where to start”

Give them this baseline rule:
- Flag a transaction if:
  - `entry_mode == online`
  - AND `device_seen_before == 0`
  - AND `amount_eur > 400`
  - OR `phish_count_24h > 0`

### “Our model flags everything / nothing”

- Ask them to inspect:
  - precision/recall curve
  - alert volume (how many per day?)
- Force a review budget:
  - e.g., “only top 2% risk scores can be flagged”

### “We can’t use message labels in test”

Correct: students receive `messages_public.csv` without `is_phishing`.

Hints:
- Use heuristics:
  - URL present + words like “verify/urgent/locked”
- Use TF‑IDF on message text and learn a phishing classifier on train messages, then apply it to test messages.
- Use LLM classification (keep prompts in write-up).

### “We don’t know how to connect messages to transactions”

Use a rolling window per user:
- count of phishing-like messages in last 24 hours
- count of suspicious keywords in last 7 days

### “We need a cost function”

Offer a simple proxy:
- Reviewing an alert costs **€2.50**
- Missing fraud costs approximately the transaction amount (use `potential_loss_eur`)
- False positive additional friction cost: **€7.50** (customer annoyance, call-center load)

### “We’re getting errors with datetime / time zones”

- Ensure `timestamp` columns are parsed:
  - `pd.to_datetime(df["timestamp"], utc=True)`
- When doing windows (e.g., last 24h), group by `user_id` and use sorted timestamps.

### “We’re worried about leakage”

Rules of thumb:
- Never compute “merchant seen before” using both train and test together.
- Prefer a time-based split in train when validating (e.g., last 15% of time for validation).
- Don’t use any test label files.

### “We want a clean submission format”

Recommend CSV:
```csv
transaction_id
T0000001
T0001234
```

Or a plain text file with one id per line.

### “We need a minimal ‘agent’ concept”

A simple “agent” is a function that:
- reads a transaction row (+ recent message features)
- returns: `{risk_score, decision, explanation, recommended_action}`

Even if the detector is a classifier, the agent packages it into a decision workflow.

## Running the sample solution (for you)

The repo includes a baseline sample you can run to generate a submission:

```bash
python3 /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/solutions/sample_solution.py --out /tmp/submission.csv
```

An “agentic” variant (adds LLM-powered alert explanations; defaults to offline mock):

```bash
LLM_MODE=mock python3 /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/solutions/agentic_solution.py \
  --out /tmp/agentic_submission.csv \
  --out-jsonl /tmp/agentic_alerts.jsonl
```

To use a real OpenAI-compatible endpoint, see `agents/README.md` for env vars.

## Optional: “investigation agent” for feature/rule ideas

If teams want an explicitly adaptive/agentic loop, you can demo:

```bash
LLM_MODE=mock python3 /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/solutions/rule_suggester.py --data-dir /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/data
```

It shows how an agent could propose new features/rules from validation error cases (or drift symptoms).

## Evaluating student submissions

Use:
- `/Users/s.zanwar/Work/Reply/NLP/TUMunternehem/eval/evaluate_submission.py`

Example:

```bash
python3 /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/eval/evaluate_submission.py \\
  --pred /path/to/team_submission.csv \\
  --data-dir /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/data/test
```

It prints:
- final score (0–100)
- F2 / precision / recall
- cost score + alert volume

## If you want to run a leaderboard locally

- Keep a folder like `submissions/TEAM_NAME.csv`
- Run the evaluator on each file and store the output JSON/CSV
- Present: `final_score`, `alerts`, `precision`, `recall`, `cost_total`

## Discussion prompts (after presentations)

- Where can drift happen? How would you detect it?
- What feedback signals exist in reality (chargebacks, analyst labels, customer calls)?
- How to prevent “gaming” (flagging too much)?
- Human-in-the-loop: which cases get reviewed first?
- Governance: privacy, fairness, auditability, model risk management
