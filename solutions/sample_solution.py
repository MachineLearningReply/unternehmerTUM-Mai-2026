#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Config:
    seed: int = 42
    msg_window_hours: int = 24
    msg_window_days: int = 7

    # Cost model (simple proxy; tune during workshop)
    false_positive_extra_cost_eur: float = 7.50
    threshold_grid: int = 199


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_time(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True)
    return out


def train_phishing_model(train_msgs: pd.DataFrame, seed: int) -> Pipeline:
    """
    Train a text classifier for phishing messages using train labels.
    """
    X = train_msgs["message_text"].fillna("").astype(str)
    y = train_msgs["is_phishing"].astype(int)
    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )
    clf.fit(X, y)
    return clf


def score_messages(msgs: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    out = msgs.copy()
    X = out["message_text"].fillna("").astype(str)
    p = model.predict_proba(X)[:, 1]
    out["phish_score"] = p
    out["is_phishing_pred"] = (p >= 0.5).astype(int)
    return out


def keyword_score(text: str) -> int:
    keywords = [
        "verify",
        "urgent",
        "locked",
        "confirm",
        "unusual",
        "blocked",
        "voucher",
        "fee",
        "password",
        "code",
        "account",
        "security",
        "pay now",
        "re-authenticate",
    ]
    if not isinstance(text, str) or not text:
        return 0
    t = text.lower()
    return int(sum(1 for k in keywords if k in t))


def build_message_features(transactions: pd.DataFrame, messages_scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Create transaction-level features from message stream per user using fixed lookback windows.
    """
    tx = transactions[["transaction_id", "user_id", "timestamp"]].copy()
    msg = messages_scored[["user_id", "timestamp", "is_phishing_pred", "phish_score", "message_text"]].copy()
    msg["kw_score"] = msg["message_text"].map(keyword_score)

    tx = tx.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    msg = msg.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    window_24h = np.timedelta64(cfg.msg_window_hours, "h")
    window_7d = np.timedelta64(cfg.msg_window_days, "D")

    feats: list[pd.DataFrame] = []
    for user_id, tx_u in tx.groupby("user_id", sort=False):
        msg_u = msg[msg["user_id"] == user_id]
        t_tx = tx_u["timestamp"].to_numpy(dtype="datetime64[ns]")
        t_msg = msg_u["timestamp"].to_numpy(dtype="datetime64[ns]")

        if len(t_msg) == 0:
            feats.append(
                pd.DataFrame(
                    {
                        "transaction_id": tx_u["transaction_id"].to_numpy(),
                        "msg_count_24h": 0,
                        "msg_count_7d": 0,
                        "phish_count_24h": 0,
                        "phish_count_7d": 0,
                        "phish_score_sum_24h": 0.0,
                        "phish_score_sum_7d": 0.0,
                        "kw_score_24h": 0,
                        "kw_score_7d": 0,
                    }
                )
            )
            continue

        is_ph = msg_u["is_phishing_pred"].to_numpy(dtype=int)
        score = msg_u["phish_score"].to_numpy(dtype=float)
        kw = msg_u["kw_score"].to_numpy(dtype=int)

        cum_msg = np.arange(1, len(t_msg) + 1)
        cum_ph = np.cumsum(is_ph)
        cum_score = np.cumsum(score)
        cum_kw = np.cumsum(kw)

        def window_aggs(window: np.timedelta64):
            right = np.searchsorted(t_msg, t_tx, side="right")
            left = np.searchsorted(t_msg, t_tx - window, side="right")

            msg_count = right - left
            ph_count = cum_ph[np.maximum(right - 1, 0)] - np.where(left > 0, cum_ph[left - 1], 0)
            score_sum = cum_score[np.maximum(right - 1, 0)] - np.where(left > 0, cum_score[left - 1], 0.0)
            kw_sum = cum_kw[np.maximum(right - 1, 0)] - np.where(left > 0, cum_kw[left - 1], 0)

            zero = right == 0
            if np.any(zero):
                ph_count = ph_count.astype(int)
                kw_sum = kw_sum.astype(int)
                ph_count[zero] = 0
                kw_sum[zero] = 0
                score_sum = score_sum.astype(float)
                score_sum[zero] = 0.0

            return msg_count.astype(int), ph_count.astype(int), score_sum.astype(float), kw_sum.astype(int)

        msg24, ph24, score24, kw24 = window_aggs(window_24h)
        msg7d, ph7d, score7d, kw7d = window_aggs(window_7d)

        feats.append(
            pd.DataFrame(
                {
                    "transaction_id": tx_u["transaction_id"].to_numpy(),
                    "msg_count_24h": msg24,
                    "msg_count_7d": msg7d,
                    "phish_count_24h": ph24,
                    "phish_count_7d": ph7d,
                    "phish_score_sum_24h": score24,
                    "phish_score_sum_7d": score7d,
                    "kw_score_24h": kw24,
                    "kw_score_7d": kw7d,
                }
            )
        )

    return pd.concat(feats, ignore_index=True)


def make_fraud_model(X: pd.DataFrame, seed: int) -> Pipeline:
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

    model = LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)
    return Pipeline(steps=[("prep", preprocess), ("model", model)])


def cost_of_decisions(df: pd.DataFrame, y_true: np.ndarray, y_hat: np.ndarray, cfg: Config) -> float:
    """
    Cost proxy:
    - Every alert has review cost (from column)
    - False positives also add friction cost
    - Missed fraud costs potential_loss_eur
    """
    review_cost = float((df["review_cost_eur"] * y_hat).sum())
    fp = (y_true == 0) & (y_hat == 1)
    fp_cost = float(fp.sum()) * cfg.false_positive_extra_cost_eur
    missed = (y_true == 1) & (y_hat == 0)
    missed_cost = float(df.loc[missed, "potential_loss_eur"].sum())
    return review_cost + fp_cost + missed_cost


def pick_threshold(df_valid: pd.DataFrame, p_valid: np.ndarray, cfg: Config) -> tuple[float, pd.DataFrame]:
    y_true = df_valid["is_fraud"].astype(int).to_numpy()
    thresholds = np.linspace(0.01, 0.99, cfg.threshold_grid)
    rows = []
    for th in thresholds:
        y_hat = (p_valid >= th).astype(int)
        cost = cost_of_decisions(df_valid, y_true=y_true, y_hat=y_hat, cfg=cfg)
        precision = 0.0 if y_hat.sum() == 0 else float(((y_true == 1) & (y_hat == 1)).sum()) / float(y_hat.sum())
        recall = 0.0 if y_true.sum() == 0 else float(((y_true == 1) & (y_hat == 1)).sum()) / float(y_true.sum())
        rows.append(
            {
                "threshold": float(th),
                "alerts": int(y_hat.sum()),
                "precision": precision,
                "recall": recall,
                "total_cost": float(cost),
            }
        )
    curve = pd.DataFrame(rows).sort_values("threshold")
    best = curve.sort_values("total_cost", ascending=True).iloc[0]
    return float(best["threshold"]), curve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sample solution: train a baseline fraud detector and create a submission.")
    parser.add_argument("--data-dir", default="data", help="Dataset root (default: data)")
    parser.add_argument("--out", default="submission.csv", help="Output submission CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    cfg = Config(seed=args.seed)
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_users = _read_csv(train_dir / "users.csv")
    train_txns = _parse_time(_read_csv(train_dir / "transactions.csv"))
    train_msgs = _parse_time(_read_csv(train_dir / "messages.csv"))

    # Public test files (no labels)
    test_tx_name = "transactions_public.csv" if (test_dir / "transactions_public.csv").exists() else "transactions.csv"
    test_msg_name = "messages_public.csv" if (test_dir / "messages_public.csv").exists() else "messages.csv"
    test_users = _read_csv(test_dir / "users.csv")
    test_txns = _parse_time(_read_csv(test_dir / test_tx_name))
    test_msgs = _parse_time(_read_csv(test_dir / test_msg_name))

    # 1) Message model (learn phishing classifier on train, apply to test)
    msg_model = train_phishing_model(train_msgs, seed=args.seed)
    train_msgs_scored = score_messages(train_msgs, msg_model)
    test_msgs_scored = score_messages(test_msgs, msg_model)

    # 2) Build transaction-level message features
    train_msg_feat = build_message_features(train_txns, train_msgs_scored, cfg)
    test_msg_feat = build_message_features(test_txns, test_msgs_scored, cfg)

    train_feat = train_txns.merge(train_msg_feat, on="transaction_id", how="left")
    test_feat = test_txns.merge(test_msg_feat, on="transaction_id", how="left")

    # 3) Time-based validation split for threshold selection
    cutoff = train_feat["timestamp"].quantile(0.85)
    train_part = train_feat[train_feat["timestamp"] <= cutoff].copy()
    valid_part = train_feat[train_feat["timestamp"] > cutoff].copy()

    drop_cols = ["is_fraud", "fraud_type", "fraud_loss_eur", "transaction_id", "timestamp"]
    X_train = train_part.drop(columns=[c for c in drop_cols if c in train_part.columns])
    y_train = train_part["is_fraud"].astype(int)
    X_valid = valid_part.drop(columns=[c for c in drop_cols if c in valid_part.columns])
    y_valid = valid_part["is_fraud"].astype(int)

    fraud_model = make_fraud_model(X_train, seed=args.seed)
    fraud_model.fit(X_train, y_train)
    p_valid = fraud_model.predict_proba(X_valid)[:, 1]

    # Quick metrics (for the organizer; harmless for students)
    try:
        print("valid ROC-AUC:", float(roc_auc_score(y_valid, p_valid)))
        print("valid PR-AUC :", float(average_precision_score(y_valid, p_valid)))
    except Exception:
        pass

    best_th, curve = pick_threshold(valid_part, p_valid, cfg)
    print("picked threshold:", best_th)
    print("alerts at threshold:", int((p_valid >= best_th).sum()))

    # 4) Refit on all train, score test, create submission
    X_all = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
    y_all = train_feat["is_fraud"].astype(int)
    fraud_model.fit(X_all, y_all)

    X_test = test_feat.drop(columns=[c for c in ["transaction_id", "timestamp"] if c in test_feat.columns])
    # Align columns (in case public test differs slightly)
    X_test = X_test.reindex(columns=X_all.columns, fill_value=np.nan)

    p_test = fraud_model.predict_proba(X_test)[:, 1]
    flagged = (p_test >= best_th).astype(int)

    submission = pd.DataFrame({"transaction_id": test_feat["transaction_id"].to_numpy()})
    submission = submission[flagged == 1].drop_duplicates().sort_values("transaction_id")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print("wrote submission:", str(out_path), "rows:", submission.shape[0])

    # Optional: create a small preview for demos
    preview_cols = [
        "timestamp",
        "user_id",
        "amount_eur",
        "merchant_name",
        "merchant_category",
        "entry_mode",
        "ip_country",
        "is_international",
        "merchant_seen_before",
        "device_seen_before",
        "phish_count_24h",
        "kw_score_24h",
    ]
    preview = test_feat.copy()
    preview["risk_score"] = p_test
    preview["flagged"] = flagged
    preview = preview[preview["flagged"] == 1].sort_values("risk_score", ascending=False).head(25)
    preview_out = out_path.with_name(out_path.stem + "_preview.csv")
    keep = [c for c in (["transaction_id"] + preview_cols + ["risk_score"]) if c in preview.columns]
    preview[keep].to_csv(preview_out, index=False)

    # Optional evaluation if labels are available (organizer-only)
    if (test_dir / "transactions_labels.csv").exists():
        labels = pd.read_csv(test_dir / "transactions_labels.csv")
        y = labels.set_index("transaction_id")["is_fraud"].astype(int)
        y_pred = pd.Series(0, index=y.index)
        y_pred.loc[submission["transaction_id"].values] = 1
        precision = float(((y_pred == 1) & (y == 1)).sum()) / max(1.0, float((y_pred == 1).sum()))
        recall = float(((y_pred == 1) & (y == 1)).sum()) / max(1.0, float((y == 1).sum()))
        print("test precision:", precision)
        print("test recall   :", recall)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

