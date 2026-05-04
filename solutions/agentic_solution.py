#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow running via: `python3 solutions/agentic_solution.py` from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.alert_agent import AlertAgent  # noqa: E402
from agents.llm import llm_from_env  # noqa: E402


@dataclass(frozen=True)
class Config:
    seed: int = 42
    threshold_grid: int = 199
    false_positive_extra_cost_eur: float = 7.50


def parse_time(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True)
    return out


def make_model(X: pd.DataFrame, seed: int) -> Pipeline:
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        ]
    )


def cost_total(df_valid: pd.DataFrame, y_true: np.ndarray, y_hat: np.ndarray, cfg: Config) -> float:
    review_cost = float((df_valid["review_cost_eur"].to_numpy(dtype=float) * y_hat).sum())
    fp = (y_true == 0) & (y_hat == 1)
    fp_cost = float(fp.sum()) * cfg.false_positive_extra_cost_eur
    missed = (y_true == 1) & (y_hat == 0)
    missed_cost = float(df_valid.loc[missed, "potential_loss_eur"].sum())
    return review_cost + fp_cost + missed_cost


def pick_threshold(valid_df: pd.DataFrame, p_valid: np.ndarray, cfg: Config) -> float:
    y_true = valid_df["is_fraud"].astype(int).to_numpy()
    thresholds = np.linspace(0.01, 0.99, cfg.threshold_grid)
    best_th = 0.5
    best_cost = float("inf")
    for th in thresholds:
        y_hat = (p_valid >= th).astype(int)
        cost = cost_total(valid_df, y_true=y_true, y_hat=y_hat, cfg=cfg)
        if cost < best_cost:
            best_cost = cost
            best_th = float(th)
    return best_th


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic sample solution: baseline model + LLM agent explanations.")
    parser.add_argument("--data-dir", default="data", help="Dataset root (default: data)")
    parser.add_argument("--out", default="submission.csv", help="Output submission CSV (transaction_id column)")
    parser.add_argument("--out-jsonl", default="alerts.jsonl", help="Output JSONL with agent explanations")
    parser.add_argument("--max-alerts", type=int, default=200, help="Max alerts to run agent over (cost control)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    cfg = Config(seed=args.seed)
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_tx = parse_time(pd.read_csv(train_dir / "transactions.csv"))
    train_msgs = parse_time(pd.read_csv(train_dir / "messages.csv"))

    test_tx_name = "transactions_public.csv" if (test_dir / "transactions_public.csv").exists() else "transactions.csv"
    test_msg_name = "messages_public.csv" if (test_dir / "messages_public.csv").exists() else "messages.csv"
    test_tx = parse_time(pd.read_csv(test_dir / test_tx_name))
    test_msgs = parse_time(pd.read_csv(test_dir / test_msg_name))

    # If public test messages have no label columns, that's fine: agent uses text and simple summaries.
    if "is_phishing_pred" not in test_msgs.columns:
        test_msgs["is_phishing_pred"] = 0

    # Train baseline model (structured features only; teams can add message-derived features).
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
    p_valid = model.predict_proba(X_valid)[:, 1]
    th = pick_threshold(valid_part, p_valid, cfg)
    print("picked threshold:", th)

    # Fit on all train and score test
    X_all = train_tx.drop(columns=[c for c in drop_cols if c in train_tx.columns])
    y_all = train_tx["is_fraud"].astype(int)
    model.fit(X_all, y_all)

    X_test = test_tx.drop(columns=[c for c in ["transaction_id", "timestamp"] if c in test_tx.columns])
    X_test = X_test.reindex(columns=X_all.columns, fill_value=np.nan)
    p_test = model.predict_proba(X_test)[:, 1]

    scored = test_tx.copy()
    scored["risk_score"] = p_test
    scored["flagged"] = (scored["risk_score"] >= th).astype(int)

    # Create CSV submission (ids only)
    submission = scored[scored["flagged"] == 1][["transaction_id"]].drop_duplicates().sort_values("transaction_id")
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_csv, index=False)
    print("wrote submission:", str(out_csv), "rows:", submission.shape[0])

    # Agentic part: generate structured explanations for (up to) top-N alerts
    llm = llm_from_env()  # default: mock
    agent = AlertAgent(llm=llm, messages=test_msgs, message_window_hours=24)

    alerts = scored[scored["flagged"] == 1].sort_values("risk_score", ascending=False).head(int(args.max_alerts)).copy()
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in alerts.iterrows():
            decision = agent.decide(row)
            record = {
                "transaction_id": row["transaction_id"],
                "user_id": row["user_id"],
                "timestamp": str(row["timestamp"]),
                "amount_eur": float(row["amount_eur"]),
                "merchant_name": str(row.get("merchant_name", "")),
                "merchant_category": str(row.get("merchant_category", "")),
                "risk_score": float(row["risk_score"]),
                "agent": {
                    "is_suspicious": int(decision.is_suspicious),
                    "severity": decision.severity,
                    "recommended_action": decision.recommended_action,
                    "explanation": decision.explanation,
                    "rationale_bullets": decision.rationale_bullets,
                    "risk_factors": decision.risk_factors,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("wrote agent alerts:", str(out_jsonl), "rows:", alerts.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
