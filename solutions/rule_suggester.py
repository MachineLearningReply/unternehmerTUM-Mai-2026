#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow running via: `python3 solutions/rule_suggester.py` from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.llm import llm_from_env  # noqa: E402
from agents.rule_agent import RuleProposalAgent  # noqa: E402


def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def make_model(X: pd.DataFrame, seed: int) -> Pipeline:
    cat = [c for c in X.columns if X[c].dtype == "object"]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )
    return Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed))])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic helper: propose new rules/features from model error cases.")
    parser.add_argument("--data-dir", default="data", help="Dataset root (default: data)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    train_tx = parse_time(pd.read_csv(Path(args.data_dir) / "train" / "transactions.csv"))
    cutoff = train_tx["timestamp"].quantile(0.85)
    train_part = train_tx[train_tx["timestamp"] <= cutoff].copy()
    valid_part = train_tx[train_tx["timestamp"] > cutoff].copy()

    drop_cols = ["is_fraud", "fraud_type", "fraud_loss_eur", "transaction_id", "timestamp"]
    X_train = train_part.drop(columns=[c for c in drop_cols if c in train_part.columns])
    y_train = train_part["is_fraud"].astype(int)
    X_valid = valid_part.drop(columns=[c for c in drop_cols if c in valid_part.columns])
    y_valid = valid_part["is_fraud"].astype(int).to_numpy()

    model = make_model(X_train, seed=args.seed)
    model.fit(X_train, y_train)
    p = model.predict_proba(X_valid)[:, 1]

    # Use a strict review budget so we intentionally create some misses (false negatives),
    # giving the agent meaningful error cases to learn from.
    review_budget = max(10, int(0.003 * len(p)))  # ~0.3% of validation
    order = np.argsort(-p)  # descending
    y_hat = np.zeros_like(y_valid, dtype=int)
    y_hat[order[:review_budget]] = 1
    th = float(p[order[review_budget - 1]]) if review_budget > 0 else 1.0

    valid_scored = valid_part.copy()
    valid_scored["risk_score"] = p
    valid_scored["pred_flagged"] = y_hat

    false_neg = valid_scored[(valid_scored["is_fraud"] == 1) & (valid_scored["pred_flagged"] == 0)].copy()
    false_pos = valid_scored[(valid_scored["is_fraud"] == 0) & (valid_scored["pred_flagged"] == 1)].copy()

    # Prioritize by cost (missed fraud loss proxy) and show some false positives for balance.
    fn_top = false_neg.sort_values("potential_loss_eur", ascending=False).head(10)
    fp_top = false_pos.sort_values("amount_eur", ascending=False).head(6)
    cases = pd.concat([fn_top, fp_top], ignore_index=True)

    cols = [
        "transaction_id",
        "timestamp",
        "amount_eur",
        "merchant_category",
        "entry_mode",
        "channel",
        "ip_country",
        "is_international",
        "merchant_seen_before",
        "device_seen_before",
        "hour",
        "is_weekend",
        "risk_score",
        "is_fraud",
        "pred_flagged",
        "potential_loss_eur",
    ]
    cases = cases[[c for c in cols if c in cases.columns]].sort_values(["is_fraud", "risk_score"], ascending=[False, False])

    llm = llm_from_env()
    agent = RuleProposalAgent(llm=llm)
    ideas = agent.propose(
        cases,
        notes="False negatives are missed fraud. False positives are benign transactions flagged. Propose improvements that help generalization and drift.",
    )

    print({"threshold_used": th, "false_negatives": int(false_neg.shape[0]), "false_positives": int(false_pos.shape[0])})
    print("\nProposed ideas:\n")
    for i, idea in enumerate(ideas.ideas, start=1):
        print(f"{i}. {idea.get('name')} ({idea.get('type')})")
        print(f"   - {idea.get('description')}")
        print(f"   - How: {idea.get('how_to_implement')}")
        print(f"   - Why: {idea.get('why_it_helps')}")
        print(f"   - Risks: {idea.get('risks')}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
