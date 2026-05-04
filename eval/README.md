# Evaluation

Students submit a list of `transaction_id`s they consider fraudulent.

This folder contains a simple evaluator that compares a submission to the hidden test labels.

## Run

```bash
python3 /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/eval/evaluate_submission.py \
  --pred /path/to/submission.csv \
  --data-dir /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/data/test
```

## Output

The script prints JSON including:
- `final_score_0_100` (weighted cost score + F2)
- `precision`, `recall`, `f2`, `tp/fp/fn/tn`
- cost breakdown (`cost_total_eur`, etc.)
- basic submission hygiene (duplicates, unknown ids)

## Scoring logic (default)

- **Cost Score (0–1)**: compares your team’s cost to:
  - baseline: flag nothing (miss all fraud)
  - oracle best: flag exactly all fraud (pay review costs, miss no fraud)
- **F2 score**: emphasizes recall (catching fraud) while penalizing low precision
- Final: `0.70 * CostScore + 0.30 * F2`, scaled to 0–100

You can change the extra false-positive penalty:

```bash
python3 /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/eval/evaluate_submission.py \
  --pred /path/to/submission.csv \
  --data-dir /Users/s.zanwar/Work/Reply/NLP/TUMunternehem/data/test \
  --fp-extra-cost 10.0
```

