# Problem Statement & Deliverables (Quick Handout)

This is a compact version of the student-facing brief.
For the full version (including suggested schedule, rules, and scoring details), use `docs/CHALLENGE_BRIEF.md`.

## Problem

Build an adaptive fraud detection prototype for banking transactions that can:

- Flag suspicious transactions under changing behavior and uncertainty
- Use both **transaction behavior** and **customer messages** (SMS/email) as signals
- Explain why a transaction was flagged (human-in-the-loop workflow)
- Choose a threshold using an explicit **cost function** (missed fraud vs false alarms)

## Deliverables

1) **Submission file**: list of suspected fraud `transaction_id`s (CSV with `transaction_id` column or text file).
2) **Write-up** (1–2 pages): approach, cost assumptions, message/text usage, limitations.
3) **Presentation** (3–5 slides): EDA insights, approach, evaluation, example alerts, next steps.
4) **Code**: notebook/script to reproduce your submission.
5) **Agentic component**: use an LLM for structured explanations and/or message understanding (document prompts + fallbacks).

## Evaluation (high-level)

- Offline detection score (cost-based + F2): 60%
- Explainability & decision workflow: 15%
- Business/risk framing: 15%
- Presentation clarity: 10%
