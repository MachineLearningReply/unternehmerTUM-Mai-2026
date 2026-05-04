# Sample Solution

This folder contains a baseline reference implementation that:

- learns a phishing text classifier from `train/messages.csv`
- builds message→transaction rolling-window features
- trains a simple fraud classifier on `train/transactions.csv`
- picks a threshold using a cost proxy
- outputs a submission file (flagged `transaction_id`s)

Run:

```bash
python3 ./solutions/sample_solution.py --out /tmp/submission.csv
```

Outputs:
- `/tmp/submission.csv`: transaction ids flagged as fraud
- `/tmp/submission_preview.csv`: top flagged examples for demo/storytelling

## Agentic variant

This variant keeps a normal fraud scoring model, but adds an LLM-powered “Alert Agent”
to produce structured explanations and recommended actions (offline mock by default):

```bash
LLM_MODE=mock python3 ./solutions/agentic_solution.py \
  --out /tmp/agentic_submission.csv \
  --out-jsonl /tmp/agentic_alerts.jsonl
```

