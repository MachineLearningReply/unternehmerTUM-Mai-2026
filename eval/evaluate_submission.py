#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoringConfig:
    # Additional penalty for false positives beyond the dataset's review_cost_eur.
    false_positive_extra_cost_eur: float = 7.50
    # Score weights (must sum to 1.0)
    w_cost: float = 0.70
    w_f2: float = 0.30
    beta: float = 2.0  # for F-beta (default: F2)


def read_prediction_ids(path: Path) -> list[str]:
    """
    Supported formats:
    - CSV with column `transaction_id`
    - Plain text: one transaction id per line
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if "transaction_id" not in df.columns:
            raise ValueError(f"CSV/TSV must contain a `transaction_id` column. Columns: {list(df.columns)}")
        ids = df["transaction_id"].astype(str).tolist()
        return ids

    # Fallback: treat as text
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("transaction_id"):
            continue
        ids.append(s)
    return ids


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a fraud submission (list of flagged transaction_ids).")
    parser.add_argument("--pred", required=True, help="Path to prediction file (CSV with transaction_id or text list)")
    parser.add_argument("--data-dir", required=True, help="Path to test data dir (contains *_public.csv and *_labels.csv)")
    parser.add_argument("--out-json", default="", help="Optional: write metrics JSON to this path")
    parser.add_argument("--fp-extra-cost", type=float, default=7.50, help="Extra cost per false positive alert (default: 7.50)")
    args = parser.parse_args(argv)

    cfg = ScoringConfig(false_positive_extra_cost_eur=float(args.fp_extra_cost))

    pred_path = Path(args.pred)
    data_dir = Path(args.data_dir)

    labels_path = data_dir / "transactions_labels.csv"
    public_path = data_dir / "transactions_public.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {labels_path}")
    if not public_path.exists():
        raise FileNotFoundError(f"Missing public transactions: {public_path}")

    labels = pd.read_csv(labels_path)
    public = pd.read_csv(public_path)
    if "transaction_id" not in labels.columns or "is_fraud" not in labels.columns:
        raise ValueError("`transactions_labels.csv` must contain columns: transaction_id, is_fraud")
    if "transaction_id" not in public.columns:
        raise ValueError("`transactions_public.csv` must contain column: transaction_id")

    df = public.merge(labels[["transaction_id", "is_fraud"]], on="transaction_id", how="left", validate="1:1")
    if df["is_fraud"].isna().any():
        raise ValueError("Labels merge failed for some rows (missing is_fraud).")

    for col in ["potential_loss_eur", "review_cost_eur"]:
        if col not in df.columns:
            raise ValueError(f"`transactions_public.csv` must include `{col}` for cost scoring.")

    df_by_id = df.set_index("transaction_id", drop=False)

    # Load predictions
    raw_ids = read_prediction_ids(pred_path)
    raw_ids = [str(x).strip() for x in raw_ids if str(x).strip()]
    submitted_count = len(raw_ids)
    unique_ids = sorted(set(raw_ids))
    duplicates = submitted_count - len(unique_ids)

    known = set(df["transaction_id"].astype(str).tolist())
    unknown_ids = [tid for tid in unique_ids if tid not in known]
    pred_ids = [tid for tid in unique_ids if tid in known]

    y_true = df_by_id["is_fraud"].astype(int)
    y_pred = pd.Series(0, index=y_true.index, dtype=int)
    if pred_ids:
        y_pred.loc[pred_ids] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = float(tp) / float(max(1, tp + fp))
    recall = float(tp) / float(max(1, tp + fn))
    f2 = fbeta(precision, recall, beta=cfg.beta)

    # Cost proxy
    y_hat = y_pred.to_numpy(dtype=int)
    y = y_true.to_numpy(dtype=int)
    review_cost = float((df_by_id["review_cost_eur"].to_numpy(dtype=float) * y_hat).sum())
    fp_cost = float(fp) * cfg.false_positive_extra_cost_eur
    missed_cost = float(df_by_id.loc[(y_true == 1) & (y_pred == 0), "potential_loss_eur"].sum())
    total_cost = review_cost + fp_cost + missed_cost

    baseline_cost = float(df_by_id.loc[y_true == 1, "potential_loss_eur"].sum())  # do nothing
    best_cost = float(df_by_id.loc[y_true == 1, "review_cost_eur"].sum())  # review exactly fraud
    denom = baseline_cost - best_cost
    cost_score = 0.0 if denom <= 0 else clamp01((baseline_cost - total_cost) / denom)

    final_score = 100.0 * (cfg.w_cost * cost_score + cfg.w_f2 * f2)
    offline_points_0_60 = 60.0 * (cfg.w_cost * cost_score + cfg.w_f2 * f2)

    metrics = {
        "final_score_0_100": final_score,
        "offline_points_0_60": offline_points_0_60,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "alerts": int((y_pred == 1).sum()),
        "alert_rate": float((y_pred == 1).mean()),
        "duplicates_in_submission": int(duplicates),
        "unknown_ids_in_submission": int(len(unknown_ids)),
        "cost_total_eur": total_cost,
        "cost_review_eur": review_cost,
        "cost_fp_extra_eur": fp_cost,
        "cost_missed_fraud_eur": missed_cost,
        "cost_baseline_do_nothing_eur": baseline_cost,
        "cost_best_oracle_eur": best_cost,
        "cost_score_0_1": cost_score,
        "scoring": {
            "false_positive_extra_cost_eur": cfg.false_positive_extra_cost_eur,
            "weights": {"cost": cfg.w_cost, "f2": cfg.w_f2},
            "beta": cfg.beta,
        },
    }

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Warn about unknown ids (don’t fail the run)
    if unknown_ids:
        sample = ", ".join(unknown_ids[:10])
        print(f"\nWARN: {len(unknown_ids)} unknown transaction_id(s) ignored. Example(s): {sample}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
