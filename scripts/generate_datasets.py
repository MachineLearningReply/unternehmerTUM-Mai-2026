#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime


def _dt(s: str) -> datetime:
    # Store timestamps in ISO-8601 with explicit UTC offset for portability.
    # (Students can localize in analysis as needed.)
    return datetime.fromisoformat(s).replace(tzinfo=timezone(timedelta(hours=1)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _choice(rng: np.random.Generator, items: list[str], p: list[float] | None = None) -> str:
    return items[int(rng.choice(len(items), p=p))]


def _random_timestamp(rng: np.random.Generator, dr: DateRange, hour_mean: float, hour_std: float) -> datetime:
    span_seconds = int((dr.end - dr.start).total_seconds())
    base = dr.start + timedelta(seconds=int(rng.integers(0, span_seconds)))
    hour = int(np.clip(rng.normal(hour_mean, hour_std), 0, 23))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return base.replace(hour=hour, minute=minute, second=second, microsecond=0)


def _hash_like_id(prefix: str, n: int, width: int = 7) -> str:
    return f"{prefix}{n:0{width}d}"


def _ip_country_for_user(rng: np.random.Generator, home_country: str, fraud: bool) -> str:
    if fraud:
        return _choice(
            rng,
            ["RO", "BG", "TR", "NG", "UA", "RU", "AE", "US", "GB"],
            p=[0.15, 0.10, 0.10, 0.10, 0.12, 0.10, 0.08, 0.15, 0.10],
        )
    # Mostly home; sometimes nearby travel/roaming.
    if rng.random() < 0.86:
        return home_country
    return _choice(rng, ["DE", "AT", "NL", "FR", "IT", "ES", "PL", "CZ"], p=[0.30, 0.10, 0.12, 0.12, 0.10, 0.08, 0.10, 0.08])


def _merchant_templates() -> dict[str, list[tuple[str, str]]]:
    # (merchant_name, merchant_country)
    return {
        "rent": [("Hausverwaltung Lindenhof", "DE"), ("Mietservice Sonnenallee", "DE"), ("Wohnbau GmbH", "DE")],
        "utilities": [("Stadtwerke Energie", "DE"), ("Wasserbetrieb Berlin", "DE"), ("Gas & Strom Direkt", "DE")],
        "phone_internet": [("TeleCom Billing", "DE"), ("NetConnect Mobile", "DE"), ("FiberHome Internet", "DE")],
        "subscription": [("StreamFlix", "IE"), ("MusicBox", "LU"), ("CloudDrive Pro", "US"), ("GymFit", "DE")],
        "groceries": [("Supermarkt König", "DE"), ("BioMarkt", "DE"), ("DiscountMart", "DE")],
        "transport": [("BVG Ticket", "DE"), ("DB Bahn", "DE"), ("RideNow", "DE")],
        "restaurants": [("Café Spreeblick", "DE"), ("Sushi Central", "DE"), ("Pizza & Pasta", "DE")],
        "electronics": [("Elektronik Markt", "DE"), ("GigaElectronics", "GB"), ("TechWorld Online", "US"), ("SmartGadgets", "NL")],
        "gift_cards": [("GiftCard Kiosk", "DE"), ("GiftCard Outlet", "US"), ("Digital Voucher Hub", "GB")],
        "crypto": [("CoinRamp", "LT"), ("CryptoNow", "EE"), ("BitTopUp", "GB")],
        "travel": [("EuroAir", "DE"), ("HotelNow", "FR"), ("RailPass", "AT"), ("CityTaxi", "DE")],
        "atm": [("ATM Withdrawal", "DE")],
        "income": [("Salary / Incoming Transfer", "DE"), ("Scholarship / Incoming Transfer", "DE")],
        "other": [("Pharmacy Plus", "DE"), ("BookStore Campus", "DE"), ("HomeGoods", "DE")],
    }


def _recurring_catalog() -> list[dict]:
    return [
        {"recurring_name": "Rent", "category": "rent", "day_of_month": 2, "amount_mean": 650, "amount_std": 80, "type": "DIRECT_DEBIT"},
        {"recurring_name": "Electricity", "category": "utilities", "day_of_month": 10, "amount_mean": 75, "amount_std": 18, "type": "DIRECT_DEBIT"},
        {"recurring_name": "Internet/Phone", "category": "phone_internet", "day_of_month": 14, "amount_mean": 40, "amount_std": 12, "type": "DIRECT_DEBIT"},
        {"recurring_name": "Streaming", "category": "subscription", "day_of_month": 20, "amount_mean": 13, "amount_std": 3, "type": "CARD"},
        {"recurring_name": "Gym", "category": "subscription", "day_of_month": 6, "amount_mean": 29, "amount_std": 6, "type": "DIRECT_DEBIT"},
    ]


def _sample_user_table(rng: np.random.Generator, n_users: int, now: datetime) -> pd.DataFrame:
    cities = ["Berlin", "Munich", "Hamburg", "Cologne", "Frankfurt", "Leipzig", "Stuttgart"]
    employment = ["student", "employed", "self_employed", "unemployed"]
    countries = ["DE", "AT", "NL", "FR", "IT", "ES", "PL", "CZ"]
    risk_profiles = ["low", "medium", "high"]
    kyc_tier = ["basic", "standard", "enhanced"]

    rows = []
    for i in range(1, n_users + 1):
        user_id = _hash_like_id("U", i, width=5)
        home_country = _choice(rng, countries, p=[0.72, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03])
        home_city = _choice(rng, cities)
        age = int(np.clip(rng.normal(29, 9), 18, 67))
        emp = _choice(rng, employment, p=[0.36, 0.48, 0.10, 0.06])
        has_2fa = int(rng.random() < 0.78)
        credit_score = int(np.clip(rng.normal(670, 60), 420, 850))
        profile = _choice(rng, risk_profiles, p=[0.44, 0.44, 0.12])
        tier = _choice(rng, kyc_tier, p=[0.30, 0.58, 0.12])

        account_age_days = int(np.clip(rng.normal(540, 380), 14, 3650))
        account_open_date = (now - timedelta(days=account_age_days)).date().isoformat()

        if emp == "student":
            income = float(np.clip(rng.normal(950, 350), 200, 2600))
            segment = "Student"
        elif emp == "unemployed":
            income = float(np.clip(rng.normal(650, 300), 0, 2200))
            segment = "Mass"
        elif emp == "self_employed":
            income = float(np.clip(rng.normal(3600, 1500), 800, 12000))
            segment = "Affluent"
        else:
            income = float(np.clip(rng.normal(2900, 1100), 700, 9000))
            segment = "Mass"

        avg_monthly_spend = float(np.clip(income * rng.uniform(0.55, 0.95), 200, 8500))
        device_count = int(np.clip(rng.normal(1.6 if has_2fa else 1.2, 0.6), 1, 4))

        rows.append(
            {
                "user_id": user_id,
                "age": age,
                "employment_status": emp,
                "segment": segment,
                "home_country": home_country,
                "home_city": home_city,
                "account_open_date": account_open_date,
                "account_age_days": account_age_days,
                "has_2fa": has_2fa,
                "device_count": device_count,
                "credit_score": credit_score,
                "risk_profile": profile,
                "kyc_tier": tier,
                "monthly_income_eur": round(income, 2),
                "avg_monthly_spend_eur": round(avg_monthly_spend, 2),
            }
        )

    return pd.DataFrame(rows)


def _generate_recurring_transactions(
    rng: np.random.Generator,
    users: pd.DataFrame,
    dr: DateRange,
    merchants: dict[str, list[tuple[str, str]]],
    start_txn_index: int,
) -> tuple[list[dict], int]:
    txns: list[dict] = []
    txn_i = start_txn_index

    recurring_catalog = _recurring_catalog()
    months = pd.period_range(dr.start.date(), dr.end.date(), freq="M")

    for _, u in users.iterrows():
        user_id = u["user_id"]
        # Not everyone has all recurring payments.
        keep = []
        for rc in recurring_catalog:
            base_keep = 0.78 if rc["recurring_name"] == "Streaming" else 0.62
            if u["employment_status"] == "student" and rc["recurring_name"] == "Gym":
                base_keep = 0.45
            if rng.random() < base_keep:
                keep.append(rc)

        # Incoming salary/scholarship
        if rng.random() < 0.85:
            incoming_name = "Salary / Incoming Transfer" if u["employment_status"] != "student" else "Scholarship / Incoming Transfer"
            for m in months:
                pay_day = int(np.clip(rng.normal(27, 2), 20, 28))
                ts = datetime(m.year, m.month, pay_day, 9, int(rng.integers(0, 60)), tzinfo=dr.start.tzinfo)
                merchant_name, merchant_country = ("Salary / Incoming Transfer", "DE") if incoming_name.startswith("Salary") else ("Scholarship / Incoming Transfer", "DE")
                amount = float(np.clip(rng.normal(u["monthly_income_eur"], u["monthly_income_eur"] * 0.05), 50, 15000))
                txns.append(
                    {
                        "transaction_id": _hash_like_id("T", txn_i),
                        "user_id": user_id,
                        "timestamp": ts.isoformat(),
                        "direction": "credit",
                        "amount_eur": round(amount, 2),
                        "currency": "EUR",
                        "transaction_type": "INCOMING",
                        "merchant_name": merchant_name,
                        "merchant_category": "income",
                        "merchant_country": merchant_country,
                        "channel": "bank_transfer",
                        "entry_mode": "not_applicable",
                        "device_id": f"{user_id}-DEV1",
                        "ip_country": u["home_country"],
                        "is_international": 0,
                        "is_recurring": 1,
                        "recurring_name": incoming_name,
                        "beneficiary_id": "none",
                        "is_fraud": 0,
                        "fraud_type": "none",
                    }
                )
                txn_i += 1

        for rc in keep:
            for m in months:
                day = min(rc["day_of_month"], int(pd.Timestamp(m.end_time).day))
                ts = datetime(m.year, m.month, day, 10, int(rng.integers(0, 60)), tzinfo=dr.start.tzinfo)
                amount = float(np.clip(rng.normal(rc["amount_mean"], rc["amount_std"]), 1.5, 2000))
                merchant_name, merchant_country = merchants[rc["category"]][int(rng.integers(0, len(merchants[rc["category"]])))]
                txns.append(
                    {
                        "transaction_id": _hash_like_id("T", txn_i),
                        "user_id": user_id,
                        "timestamp": ts.isoformat(),
                        "direction": "debit",
                        "amount_eur": round(amount, 2),
                        "currency": "EUR",
                        "transaction_type": rc["type"],
                        "merchant_name": merchant_name,
                        "merchant_category": rc["category"],
                        "merchant_country": merchant_country,
                        "channel": "card" if rc["type"] == "CARD" else "direct_debit",
                        "entry_mode": "chip" if rc["type"] == "CARD" else "not_applicable",
                        "device_id": f"{user_id}-DEV1",
                        "ip_country": u["home_country"],
                        "is_international": int(merchant_country != u["home_country"]),
                        "is_recurring": 1,
                        "recurring_name": rc["recurring_name"],
                        "beneficiary_id": "none",
                        "is_fraud": 0,
                        "fraud_type": "none",
                    }
                )
                txn_i += 1

    return txns, txn_i


def _generate_everyday_transactions(
    rng: np.random.Generator,
    users: pd.DataFrame,
    dr: DateRange,
    merchants: dict[str, list[tuple[str, str]]],
    start_txn_index: int,
) -> tuple[list[dict], int]:
    txns: list[dict] = []
    txn_i = start_txn_index

    base_categories = [
        "groceries",
        "transport",
        "restaurants",
        "other",
        "subscription",
        "utilities",
        "electronics",
        "travel",
        "gift_cards",
        "crypto",
    ]

    for _, u in users.iterrows():
        user_id = u["user_id"]
        home_country = u["home_country"]
        home_city = u["home_city"]

        # Behavioral priors per user:
        hour_mean = float(np.clip(rng.normal(16.5, 2.0), 11.0, 21.0))
        hour_std = float(np.clip(rng.normal(3.2, 0.8), 1.5, 5.5))
        weekly_rate = float(np.clip(rng.normal(13, 4), 5, 26))  # avg transactions per week

        n_days = (dr.end.date() - dr.start.date()).days + 1
        expected = weekly_rate / 7.0 * n_days
        n_txns = int(np.clip(rng.normal(expected, expected * 0.15), expected * 0.6, expected * 1.5))

        user_devices = [f"{user_id}-DEV1"]
        for d in range(2, int(u["device_count"]) + 1):
            user_devices.append(f"{user_id}-DEV{d}")

        for _ in range(n_txns):
            category = _choice(
                rng,
                base_categories,
                p=[0.24, 0.16, 0.15, 0.14, 0.10, 0.10, 0.06, 0.03, 0.01, 0.01],
            )
            merchant_name, merchant_country = merchants[category][int(rng.integers(0, len(merchants[category])))]
            ts = _random_timestamp(rng, dr, hour_mean=hour_mean, hour_std=hour_std)

            # Some legitimate travel/international activity (not fraud by itself).
            is_international = int(merchant_country != home_country and rng.random() < (0.75 if category in {"travel", "electronics"} else 0.55))
            if is_international and category != "atm":
                # Make country more "travel-realistic".
                merchant_country = _choice(rng, ["DE", "AT", "NL", "FR", "IT", "ES", "CH", "GB"], p=[0.15, 0.12, 0.12, 0.12, 0.12, 0.10, 0.12, 0.15])

            if category == "groceries":
                amount = float(np.clip(rng.normal(32, 18), 3.5, 190))
            elif category == "transport":
                amount = float(np.clip(rng.normal(16, 12), 1.8, 220))
            elif category == "restaurants":
                amount = float(np.clip(rng.normal(24, 18), 4.0, 260))
            elif category == "utilities":
                amount = float(np.clip(rng.normal(22, 20), 2.0, 320))
            elif category == "subscription":
                amount = float(np.clip(rng.normal(11, 8), 2.5, 120))
            elif category == "electronics":
                # Legitimate electronics purchases exist; fraud scenarios use additional context (new device, phishing, testing, odd hours).
                amount = float(np.clip(rng.normal(180, 220), 8.0, 2400))
            elif category == "travel":
                amount = float(np.clip(rng.normal(120, 180), 6.0, 1800))
            elif category == "gift_cards":
                amount = float(np.clip(rng.normal(35, 30), 5.0, 250))
            elif category == "crypto":
                amount = float(np.clip(rng.normal(110, 140), 10.0, 1200))
            else:
                amount = float(np.clip(rng.normal(38, 40), 2.0, 420))

            txn_type = _choice(rng, ["CARD", "TRANSFER", "ATM"], p=[0.76, 0.16, 0.08])
            channel = "card" if txn_type == "CARD" else ("bank_transfer" if txn_type == "TRANSFER" else "atm")
            entry_mode = _choice(rng, ["chip", "contactless", "online"], p=[0.25, 0.55, 0.20]) if txn_type == "CARD" else "not_applicable"

            beneficiary_id = ""
            merchant_cat = category
            merchant = merchant_name
            if txn_type == "ATM":
                merchant_cat = "atm"
                merchant, merchant_country = merchants["atm"][0]
                amount = float(np.clip(rng.normal(70, 50), 20, 600))
                entry_mode = "not_applicable"

            if txn_type == "TRANSFER":
                merchant_cat = "other"
                merchant = _choice(rng, ["Family Transfer", "Rent Top-up", "Marketplace Seller", "Shared Expenses"], p=[0.30, 0.20, 0.22, 0.28])
                beneficiary_id = _hash_like_id("B", int(rng.integers(1, 5000)), width=6)
                amount = float(np.clip(rng.normal(95, 120), 8, 1400))
                entry_mode = "not_applicable"

            device_id = _choice(rng, user_devices)
            ip_country = _ip_country_for_user(rng, home_country, fraud=False)

            txns.append(
                {
                    "transaction_id": _hash_like_id("T", txn_i),
                    "user_id": user_id,
                    "timestamp": ts.isoformat(),
                    "direction": "debit",
                    "amount_eur": round(amount, 2),
                    "currency": "EUR",
                    "transaction_type": txn_type,
                    "merchant_name": merchant,
                    "merchant_category": merchant_cat,
                    "merchant_country": merchant_country,
                    "channel": channel,
                    "entry_mode": entry_mode,
                    "device_id": device_id,
                    "ip_country": ip_country,
                    "is_international": int(merchant_country != home_country),
                    "is_recurring": 0,
                    "recurring_name": "none",
                    "beneficiary_id": beneficiary_id,
                    "is_fraud": 0,
                    "fraud_type": "none",
                    "user_home_city": home_city,
                }
            )
            txn_i += 1

    return txns, txn_i


def _generate_messages_for_user(
    rng: np.random.Generator,
    user_id: str,
    dr: DateRange,
    base_rate_per_week: float,
    message_index_start: int,
) -> tuple[list[dict], int]:
    messages: list[dict] = []
    msg_i = message_index_start

    n_days = (dr.end.date() - dr.start.date()).days + 1
    expected = base_rate_per_week / 7.0 * n_days
    n_msgs = int(np.clip(rng.normal(expected, expected * 0.25), expected * 0.5, expected * 1.6))

    benign_bank = [
        ("Bank Alerts", "SMS", "Your account statement is ready in the app."),
        ("Bank Alerts", "SMS", "Security tip: Never share your one-time codes."),
        ("Bank Alerts", "SMS", "Reminder: Update your app to the latest version for improved security."),
    ]
    marketing = [
        ("ShopPromo", "SMS", "Flash sale today: 20% off selected items. Unsubscribe: STOP"),
        ("TravelDeals", "Email", "Weekend deals: Save on hotels in Europe."),
        ("CampusEvents", "Email", "Student event: Networking meetup next Thursday."),
    ]
    phishing = [
        ("Bank-Security", "SMS", "Unusual activity detected. To keep your account active, verify at https://bank-secure-check.example"),
        ("Delivery-Notice", "SMS", "Your parcel is held due to unpaid customs fee. Pay now: https://track-fee.example"),
        ("IT-Support", "Email", "Password expires today. Re-authenticate here: https://sso-reset.example"),
        ("Card-Verify", "SMS", "Card verification required. Confirm now to avoid lock: https://verify-card.example"),
    ]

    for _ in range(n_msgs):
        ts = _random_timestamp(rng, dr, hour_mean=float(np.clip(rng.normal(13.5, 2.0), 8, 20)), hour_std=2.8)
        kind_roll = rng.random()
        if kind_roll < 0.55:
            sender, channel, text = benign_bank[int(rng.integers(0, len(benign_bank)))]
            is_phish = 0
        elif kind_roll < 0.85:
            sender, channel, text = marketing[int(rng.integers(0, len(marketing)))]
            is_phish = 0
        else:
            sender, channel, text = phishing[int(rng.integers(0, len(phishing)))]
            is_phish = 1

        messages.append(
            {
                "message_id": _hash_like_id("M", msg_i, width=7),
                "user_id": user_id,
                "timestamp": ts.isoformat(),
                "channel": channel,
                "sender": sender,
                "message_text": text,
                "is_phishing": is_phish,
            }
        )
        msg_i += 1

    return messages, msg_i


def _inject_fraud_scenarios(
    rng: np.random.Generator,
    users: pd.DataFrame,
    dr: DateRange,
    merchants: dict[str, list[tuple[str, str]]],
    txns: list[dict],
    messages: list[dict],
    start_message_index: int,
    fraud_mix: dict[str, float],
    hard_mode: bool,
) -> int:
    """Inject fraud + correlated messages. Mutates `txns` and `messages` in-place."""
    msg_i = start_message_index
    txn_counter = max((int(t["transaction_id"][1:]) for t in txns), default=0)

    def new_txn_id() -> str:
        nonlocal txn_counter
        txn_counter += 1
        return _hash_like_id("T", txn_counter)

    users_by_id = {row["user_id"]: row for _, row in users.iterrows()}

    known_merchants: dict[str, set[str]] = {}
    for t in txns:
        known_merchants.setdefault(t["user_id"], set()).add(t["merchant_name"])

    # Workshop-friendly fraud prevalence: higher than real-world to ensure enough positives.
    fraud_user_count = int(max(12, round(len(users) * (0.28 if not hard_mode else 0.32))))
    fraud_users = rng.choice(users["user_id"].to_numpy(), size=fraud_user_count, replace=False)

    scenario_names = list(fraud_mix.keys())
    scenario_probs = np.array([fraud_mix[k] for k in scenario_names], dtype=float)
    scenario_probs = scenario_probs / scenario_probs.sum()

    for user_id in fraud_users:
        u = users_by_id[user_id]
        home_country = u["home_country"]
        typical_device = f"{user_id}-DEV1"
        new_device = f"{user_id}-DEVX"

        # Some customers get repeated attempts.
        scenario_runs = 2 if rng.random() < (0.22 if hard_mode else 0.16) else 1
        for _ in range(scenario_runs):
            scenario_ts = _random_timestamp(
                rng,
                dr,
                hour_mean=float(np.clip(rng.normal(2.4, 3.0), 0, 23)),
                hour_std=float(np.clip(rng.normal(2.0, 1.2), 0.8, 6.0)),
            )
            scenario = _choice(rng, scenario_names, p=scenario_probs.tolist())

            user_known = known_merchants.setdefault(user_id, set())

            if scenario == "card_testing_then_big_purchase":
                test_merchant = "ZX*PAYMENT TEST"
                for amt in [1.00, 1.50, 2.00]:
                    txns.append(
                        {
                            "transaction_id": new_txn_id(),
                            "user_id": user_id,
                            "timestamp": (scenario_ts - timedelta(minutes=int(rng.integers(20, 75)))).isoformat(),
                            "direction": "debit",
                            "amount_eur": amt,
                            "currency": "EUR",
                            "transaction_type": "CARD",
                            "merchant_name": test_merchant,
                            "merchant_category": "electronics",
                            "merchant_country": _choice(rng, ["GB", "US", "NL"], p=[0.45, 0.35, 0.20]),
                            "channel": "card",
                            "entry_mode": "online",
                            "device_id": new_device,
                            "ip_country": _ip_country_for_user(rng, home_country, fraud=True),
                            "is_international": 1,
                            "is_recurring": 0,
                            "recurring_name": "none",
                            "beneficiary_id": "none",
                            "is_fraud": 1,
                            "fraud_type": "card_testing",
                            "user_home_city": u["home_city"],
                        }
                    )
                    user_known.add(test_merchant)

                merchant_name, merchant_country = merchants["electronics"][int(rng.integers(0, len(merchants["electronics"])))]
                big_amt = float(np.clip(rng.normal(980, 420), 260, 2600))
                chosen_merchant = merchant_name if merchant_name not in user_known else "GigaElectronics"
                txns.append(
                    {
                        "transaction_id": new_txn_id(),
                        "user_id": user_id,
                        "timestamp": (scenario_ts + timedelta(minutes=int(rng.integers(1, 25)))).isoformat(),
                        "direction": "debit",
                        "amount_eur": round(big_amt, 2),
                        "currency": "EUR",
                        "transaction_type": "CARD",
                        "merchant_name": chosen_merchant,
                        "merchant_category": "electronics",
                        "merchant_country": merchant_country,
                        "channel": "card",
                        "entry_mode": "online",
                        "device_id": new_device,
                        "ip_country": _ip_country_for_user(rng, home_country, fraud=True),
                        "is_international": 1,
                        "is_recurring": 0,
                        "recurring_name": "none",
                        "beneficiary_id": "none",
                        "is_fraud": 1,
                        "fraud_type": "card_not_present",
                        "user_home_city": u["home_city"],
                    }
                )
                user_known.add(chosen_merchant)

                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts + timedelta(minutes=8)).isoformat(),
                        "channel": "SMS",
                        "sender": "Bank Alerts",
                        "message_text": f"Security alert: Card payment of {round(big_amt,2)} EUR at {chosen_merchant} detected. If this wasn't you, call support.",
                        "is_phishing": 0,
                    }
                )
                msg_i += 1

            elif scenario == "account_takeover_transfer":
                otp = int(rng.integers(100000, 999999))
                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts - timedelta(minutes=18)).isoformat(),
                        "channel": "SMS",
                        "sender": "Bank-Security",
                        "message_text": "Unusual activity detected. Verify now: https://bank-secure-check.example",
                        "is_phishing": 1,
                    }
                )
                msg_i += 1
                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts - timedelta(minutes=7)).isoformat(),
                        "channel": "SMS",
                        "sender": "Bank Alerts",
                        "message_text": f"Your one-time code is {otp}. Do not share this code with anyone.",
                        "is_phishing": 0,
                    }
                )
                msg_i += 1

                amt = float(np.clip(rng.normal(2150, 950), 600, 5500))
                beneficiary_id = _hash_like_id("B", int(rng.integers(8000, 14000)), width=6)
                txns.append(
                    {
                        "transaction_id": new_txn_id(),
                        "user_id": user_id,
                        "timestamp": scenario_ts.isoformat(),
                        "direction": "debit",
                        "amount_eur": round(amt, 2),
                        "currency": "EUR",
                        "transaction_type": "TRANSFER",
                        "merchant_name": _choice(rng, ["SafeAccount GmbH", "SecurePay Services", "Urgent Invoice Settlement"], p=[0.40, 0.35, 0.25]),
                        "merchant_category": "other",
                        "merchant_country": home_country,
                        "channel": "bank_transfer",
                        "entry_mode": "not_applicable",
                        "device_id": new_device,
                        "ip_country": _ip_country_for_user(rng, home_country, fraud=True),
                        "is_international": 0,
                        "is_recurring": 0,
                        "recurring_name": "none",
                        "beneficiary_id": beneficiary_id,
                        "is_fraud": 1,
                        "fraud_type": "account_takeover",
                        "user_home_city": u["home_city"],
                    }
                )

                if rng.random() < 0.6:
                    atm_amt = float(np.clip(rng.normal(420, 180), 120, 900))
                    txns.append(
                        {
                            "transaction_id": new_txn_id(),
                            "user_id": user_id,
                            "timestamp": (scenario_ts + timedelta(minutes=int(rng.integers(12, 38)))).isoformat(),
                            "direction": "debit",
                            "amount_eur": round(atm_amt, 2),
                            "currency": "EUR",
                            "transaction_type": "ATM",
                            "merchant_name": "ATM Withdrawal",
                            "merchant_category": "atm",
                            "merchant_country": home_country,
                            "channel": "atm",
                            "entry_mode": "not_applicable",
                            "device_id": new_device,
                            "ip_country": _ip_country_for_user(rng, home_country, fraud=True),
                            "is_international": 0,
                            "is_recurring": 0,
                            "recurring_name": "none",
                            "beneficiary_id": "none",
                            "is_fraud": 1,
                            "fraud_type": "cash_out",
                            "user_home_city": u["home_city"],
                        }
                    )

                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts + timedelta(minutes=2)).isoformat(),
                        "channel": "SMS",
                        "sender": "Bank Alerts",
                        "message_text": "New device sign-in detected. If this wasn't you, change your password immediately.",
                        "is_phishing": 0,
                    }
                )
                msg_i += 1

            elif scenario == "invoice_manipulation":
                bill_amt = float(np.clip(rng.normal(92, 22), 35, 180))
                beneficiary_id = _hash_like_id("B", int(rng.integers(14000, 19000)), width=6)

                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts - timedelta(hours=3)).isoformat(),
                        "channel": "Email",
                        "sender": "billing@stadtwerke-energie.example",
                        "message_text": "Invoice update: Due to a system change, please use the new bank details in the attached invoice PDF.",
                        "is_phishing": 1,
                    }
                )
                msg_i += 1

                txns.append(
                    {
                        "transaction_id": new_txn_id(),
                        "user_id": user_id,
                        "timestamp": scenario_ts.isoformat(),
                        "direction": "debit",
                        "amount_eur": round(bill_amt, 2),
                        "currency": "EUR",
                        "transaction_type": "TRANSFER",
                        "merchant_name": _choice(rng, ["Stadtwerke Enerqie", "Stadtwerke Energie Service", "Energy Billing Update"], p=[0.40, 0.35, 0.25]),
                        "merchant_category": "utilities",
                        "merchant_country": home_country,
                        "channel": "bank_transfer",
                        "entry_mode": "not_applicable",
                        "device_id": typical_device if rng.random() < 0.65 else new_device,
                        "ip_country": _ip_country_for_user(rng, home_country, fraud=(rng.random() < 0.65)),
                        "is_international": 0,
                        "is_recurring": 0,
                        "recurring_name": "none",
                        "beneficiary_id": beneficiary_id,
                        "is_fraud": 1,
                        "fraud_type": "invoice_redirection",
                        "user_home_city": u["home_city"],
                    }
                )

            elif scenario == "gift_cards_social_engineering":
                base = float(np.clip(rng.normal(180, 60), 60, 320))
                use_device = typical_device if rng.random() < 0.15 else new_device
                use_ip_fraud = rng.random() < 0.85
                for k in range(int(rng.integers(2, 5))):
                    amt = round(base + float(rng.normal(0, 18)), 2)
                    merchant_name, merchant_country = merchants["gift_cards"][int(rng.integers(0, len(merchants["gift_cards"])))]
                    txns.append(
                        {
                            "transaction_id": new_txn_id(),
                            "user_id": user_id,
                            "timestamp": (scenario_ts + timedelta(minutes=7 * k)).isoformat(),
                            "direction": "debit",
                            "amount_eur": amt,
                            "currency": "EUR",
                            "transaction_type": "CARD",
                            "merchant_name": merchant_name,
                            "merchant_category": "gift_cards",
                            "merchant_country": merchant_country,
                            "channel": "card",
                            "entry_mode": "online",
                            "device_id": use_device,
                            "ip_country": _ip_country_for_user(rng, home_country, fraud=use_ip_fraud),
                            "is_international": int(merchant_country != home_country),
                            "is_recurring": 0,
                            "recurring_name": "none",
                            "beneficiary_id": "none",
                            "is_fraud": 1,
                            "fraud_type": "social_engineering",
                            "user_home_city": u["home_city"],
                        }
                    )
                    user_known.add(merchant_name)

                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts - timedelta(minutes=12)).isoformat(),
                        "channel": "SMS",
                        "sender": "IT-Support",
                        "message_text": "Urgent: Your corporate account is locked. Buy 3 digital vouchers and send the codes to re-enable access.",
                        "is_phishing": 1,
                    }
                )
                msg_i += 1

            elif scenario == "crypto_topup_drift":
                repeats = int(rng.integers(2, 6))
                for k in range(repeats):
                    merchant_name, merchant_country = merchants["crypto"][int(rng.integers(0, len(merchants["crypto"])))]
                    amt = float(np.clip(rng.normal(220, 120), 40, 600))
                    if k == repeats - 1:
                        amt = float(np.clip(rng.normal(980, 420), 220, 2600))
                    txns.append(
                        {
                            "transaction_id": new_txn_id(),
                            "user_id": user_id,
                            "timestamp": (scenario_ts + timedelta(minutes=11 * k)).isoformat(),
                            "direction": "debit",
                            "amount_eur": round(amt, 2),
                            "currency": "EUR",
                            "transaction_type": "TRANSFER" if rng.random() < 0.55 else "CARD",
                            "merchant_name": merchant_name,
                            "merchant_category": "crypto",
                            "merchant_country": merchant_country,
                            "channel": "bank_transfer" if rng.random() < 0.55 else "card",
                            "entry_mode": "online",
                            "device_id": new_device,
                            "ip_country": _ip_country_for_user(rng, home_country, fraud=True),
                            "is_international": int(merchant_country != home_country),
                            "is_recurring": 0,
                            "recurring_name": "none",
                            "beneficiary_id": _hash_like_id("B", int(rng.integers(19000, 24000)), width=6),
                            "is_fraud": 1,
                            "fraud_type": "crypto_ramp",
                            "user_home_city": u["home_city"],
                        }
                    )
                    user_known.add(merchant_name)

                messages.append(
                    {
                        "message_id": _hash_like_id("M", msg_i, width=7),
                        "user_id": user_id,
                        "timestamp": (scenario_ts - timedelta(minutes=25)).isoformat(),
                        "channel": "SMS",
                        "sender": "Delivery-Notice",
                        "message_text": "A package is waiting. Confirm address and pay small fee: https://track-fee.example",
                        "is_phishing": 1,
                    }
                )
                msg_i += 1

    return msg_i


def _inject_legit_lookalikes(
    rng: np.random.Generator,
    users: pd.DataFrame,
    dr: DateRange,
    merchants: dict[str, list[tuple[str, str]]],
    txns: list[dict],
    messages: list[dict],
    start_message_index: int,
    hard_mode: bool,
) -> int:
    """
    Inject legitimate edge cases that look suspicious to reduce trivial separability.
    Mutates txns/messages in-place.
    """
    msg_i = start_message_index
    txn_counter = max((int(t["transaction_id"][1:]) for t in txns), default=0)

    def new_txn_id() -> str:
        nonlocal txn_counter
        txn_counter += 1
        return _hash_like_id("T", txn_counter)

    users_by_id = {row["user_id"]: row for _, row in users.iterrows()}

    lookalike_user_count = int(max(10, round(len(users) * (0.10 if not hard_mode else 0.12))))
    look_users = rng.choice(users["user_id"].to_numpy(), size=lookalike_user_count, replace=False)

    for user_id in look_users:
        u = users_by_id[user_id]
        home_country = u["home_country"]
        typical_device = f"{user_id}-DEV1"
        alt_device = f"{user_id}-DEV2"

        ts = _random_timestamp(rng, dr, hour_mean=float(np.clip(rng.normal(18.0, 2.5), 8, 23)), hour_std=2.0)
        scenario = _choice(rng, ["legit_big_electronics", "legit_gift_cards", "legit_crypto", "legit_travel_spike"], p=[0.30, 0.26, 0.22, 0.22])

        if scenario == "legit_big_electronics":
            merchant_name, merchant_country = merchants["electronics"][int(rng.integers(0, len(merchants["electronics"])))]
            amt = float(np.clip(rng.normal(950, 380), 350, 2600))
            txns.append(
                {
                    "transaction_id": new_txn_id(),
                    "user_id": user_id,
                    "timestamp": ts.isoformat(),
                    "direction": "debit",
                    "amount_eur": round(amt, 2),
                    "currency": "EUR",
                    "transaction_type": "CARD",
                    "merchant_name": merchant_name,
                    "merchant_category": "electronics",
                    "merchant_country": merchant_country,
                    "channel": "card",
                    "entry_mode": "online" if rng.random() < 0.65 else "chip",
                    "device_id": alt_device if rng.random() < 0.55 else typical_device,
                    "ip_country": _ip_country_for_user(rng, home_country, fraud=False),
                    "is_international": int(merchant_country != home_country),
                    "is_recurring": 0,
                    "recurring_name": "none",
                    "beneficiary_id": "none",
                    "is_fraud": 0,
                    "fraud_type": "none",
                    "user_home_city": u["home_city"],
                }
            )
            messages.append(
                {
                    "message_id": _hash_like_id("M", msg_i, width=7),
                    "user_id": user_id,
                    "timestamp": (ts + timedelta(minutes=4)).isoformat(),
                    "channel": "SMS",
                    "sender": "Bank Alerts",
                    "message_text": f"Info: You made a card payment of {round(amt,2)} EUR at {merchant_name}. If correct, no action needed.",
                    "is_phishing": 0,
                }
            )
            msg_i += 1

        elif scenario == "legit_gift_cards":
            base = float(np.clip(rng.normal(95, 25), 30, 160))
            for k in range(int(rng.integers(2, 4))):
                merchant_name, merchant_country = merchants["gift_cards"][0]  # DE kiosk
                amt = round(base + float(rng.normal(0, 8)), 2)
                txns.append(
                    {
                        "transaction_id": new_txn_id(),
                        "user_id": user_id,
                        "timestamp": (ts + timedelta(minutes=9 * k)).isoformat(),
                        "direction": "debit",
                        "amount_eur": amt,
                        "currency": "EUR",
                        "transaction_type": "CARD",
                        "merchant_name": merchant_name,
                        "merchant_category": "gift_cards",
                        "merchant_country": merchant_country,
                        "channel": "card",
                        "entry_mode": "contactless" if rng.random() < 0.6 else "online",
                        "device_id": alt_device if rng.random() < 0.35 else typical_device,
                        "ip_country": _ip_country_for_user(rng, home_country, fraud=False),
                        "is_international": 0,
                        "is_recurring": 0,
                        "recurring_name": "none",
                        "beneficiary_id": "none",
                        "is_fraud": 0,
                        "fraud_type": "none",
                        "user_home_city": u["home_city"],
                    }
                )

            messages.append(
                {
                    "message_id": _hash_like_id("M", msg_i, width=7),
                    "user_id": user_id,
                    "timestamp": (ts - timedelta(minutes=20)).isoformat(),
                    "channel": "SMS",
                    "sender": "ShopPromo",
                    "message_text": "Reminder: Gift cards purchased in-store are non-refundable. Keep your receipt.",
                    "is_phishing": 0,
                }
            )
            msg_i += 1

        elif scenario == "legit_crypto":
            repeats = int(rng.integers(1, 4))
            for k in range(repeats):
                merchant_name, merchant_country = merchants["crypto"][int(rng.integers(0, len(merchants["crypto"])))]
                amt = float(np.clip(rng.normal(160, 110), 30, 650))
                txns.append(
                    {
                        "transaction_id": new_txn_id(),
                        "user_id": user_id,
                        "timestamp": (ts + timedelta(minutes=17 * k)).isoformat(),
                        "direction": "debit",
                        "amount_eur": round(amt, 2),
                        "currency": "EUR",
                        "transaction_type": "TRANSFER" if rng.random() < 0.65 else "CARD",
                        "merchant_name": merchant_name,
                        "merchant_category": "crypto",
                        "merchant_country": merchant_country,
                        "channel": "bank_transfer" if rng.random() < 0.65 else "card",
                        "entry_mode": "online",
                        "device_id": typical_device,
                        "ip_country": _ip_country_for_user(rng, home_country, fraud=False),
                        "is_international": int(merchant_country != home_country),
                        "is_recurring": 0,
                        "recurring_name": "none",
                        "beneficiary_id": _hash_like_id("B", int(rng.integers(24000, 28000)), width=6),
                        "is_fraud": 0,
                        "fraud_type": "none",
                        "user_home_city": u["home_city"],
                    }
                )

        else:  # legit_travel_spike
            # A short burst of international spending while traveling.
            for k in range(int(rng.integers(3, 7))):
                merchant_name, merchant_country = merchants["travel"][int(rng.integers(0, len(merchants["travel"])))]
                amt = float(np.clip(rng.normal(85, 70), 6, 800))
                txns.append(
                    {
                        "transaction_id": new_txn_id(),
                        "user_id": user_id,
                        "timestamp": (ts + timedelta(hours=int(rng.integers(0, 36)), minutes=int(rng.integers(0, 60)))).isoformat(),
                        "direction": "debit",
                        "amount_eur": round(amt, 2),
                        "currency": "EUR",
                        "transaction_type": "CARD",
                        "merchant_name": merchant_name,
                        "merchant_category": "travel",
                        "merchant_country": merchant_country,
                        "channel": "card",
                        "entry_mode": "contactless",
                        "device_id": alt_device if rng.random() < 0.5 else typical_device,
                        "ip_country": merchant_country,
                        "is_international": int(merchant_country != home_country),
                        "is_recurring": 0,
                        "recurring_name": "none",
                        "beneficiary_id": "none",
                        "is_fraud": 0,
                        "fraud_type": "none",
                        "user_home_city": u["home_city"],
                    }
                )

            messages.append(
                {
                    "message_id": _hash_like_id("M", msg_i, width=7),
                    "user_id": user_id,
                    "timestamp": (ts - timedelta(hours=2)).isoformat(),
                    "channel": "Email",
                    "sender": "TravelDeals",
                    "message_text": "Trip reminder: Keep your card with you and enable roaming notifications in your banking app.",
                    "is_phishing": 0,
                }
            )
            msg_i += 1

    return msg_i


def _postprocess_transactions(txns: list[dict], users: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(txns)
    if "user_home_city" not in df.columns:
        df["user_home_city"] = ""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Enrich with user-level fields for convenience (still keep users.csv as source of truth).
    users_small = users[["user_id", "home_country", "home_city", "employment_status", "segment", "has_2fa", "credit_score", "risk_profile"]].copy()
    df = df.merge(users_small, on="user_id", how="left", validate="m:1")

    # User-merchant familiarity features (computed from history inside the split).
    df = df.sort_values(["user_id", "timestamp", "transaction_id"]).reset_index(drop=True)
    df["merchant_seen_before"] = 0
    df["device_seen_before"] = 0
    seen_merchants: dict[str, set[str]] = {}
    seen_devices: dict[str, set[str]] = {}

    for idx, row in df.iterrows():
        uid = row["user_id"]
        m = row["merchant_name"]
        d = row["device_id"]
        if uid not in seen_merchants:
            seen_merchants[uid] = set()
            seen_devices[uid] = set()
        df.at[idx, "merchant_seen_before"] = int(m in seen_merchants[uid])
        df.at[idx, "device_seen_before"] = int(d in seen_devices[uid])
        seen_merchants[uid].add(m)
        seen_devices[uid].add(d)

    # A simple "risk hint" that is not a label, but captures asymmetric costs:
    # higher transaction amount -> higher potential loss.
    df["potential_loss_eur"] = np.where(df["direction"] == "debit", df["amount_eur"], 0.0)
    df["review_cost_eur"] = 2.50  # operational cost of manual review
    df["fraud_loss_eur"] = np.where(df["is_fraud"].astype(int) == 1, df["potential_loss_eur"], 0.0)

    # Normalize types
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["is_recurring"] = df["is_recurring"].astype(int)
    df["is_international"] = df["is_international"].astype(int)

    # Keep deterministic, stable IDs ordering.
    df["transaction_id_num"] = df["transaction_id"].str[1:].astype(int)
    df = df.sort_values("transaction_id_num").drop(columns=["transaction_id_num"]).reset_index(drop=True)

    # Convert timestamp back to ISO strings for CSV portability.
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    # Insert colon in offset: +0100 -> +01:00
    df["timestamp"] = df["timestamp"].str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
    return df


def _postprocess_messages(messages: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(messages)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp", "message_id"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["is_phishing"] = df["is_phishing"].astype(int)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    df["timestamp"] = df["timestamp"].str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
    return df


def _build_split(seed: int, n_users: int, dr: DateRange, fraud_mix: dict[str, float], hard_mode: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    merchants = _merchant_templates()
    now = dr.end
    users = _sample_user_table(rng, n_users=n_users, now=now)

    txns: list[dict] = []
    txn_i = 1
    recurring_txns, txn_i = _generate_recurring_transactions(rng, users, dr, merchants, start_txn_index=txn_i)
    txns.extend(recurring_txns)
    everyday_txns, txn_i = _generate_everyday_transactions(rng, users, dr, merchants, start_txn_index=txn_i)
    txns.extend(everyday_txns)

    messages: list[dict] = []
    msg_i = 1
    # Base messages per user
    for _, u in users.iterrows():
        base_msgs, msg_i = _generate_messages_for_user(
            rng,
            user_id=u["user_id"],
            dr=dr,
            base_rate_per_week=float(np.clip(rng.normal(3.8, 1.1), 1.2, 7.8)),
            message_index_start=msg_i,
        )
        messages.extend(base_msgs)

    # Inject fraud + correlated messages
    msg_i = _inject_fraud_scenarios(
        rng,
        users=users,
        dr=dr,
        merchants=merchants,
        txns=txns,
        messages=messages,
        start_message_index=msg_i,
        fraud_mix=fraud_mix,
        hard_mode=hard_mode,
    )

    # Inject legitimate “lookalikes” (suspicious but non-fraud) to avoid trivial separability.
    msg_i = _inject_legit_lookalikes(
        rng,
        users=users,
        dr=dr,
        merchants=merchants,
        txns=txns,
        messages=messages,
        start_message_index=msg_i,
        hard_mode=hard_mode,
    )

    # Postprocess
    transactions = _postprocess_transactions(txns, users)
    messages_df = _postprocess_messages(messages)

    # Sort users stable
    users = users.sort_values("user_id").reset_index(drop=True)
    return users, transactions, messages_df


def write_split(out_dir: str, users: pd.DataFrame, transactions: pd.DataFrame, messages: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    users.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    transactions.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
    messages.to_csv(os.path.join(out_dir, "messages.csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud-detection workshop datasets (train/test).")
    parser.add_argument("--out", default="data", help="Output directory (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    train_range = DateRange(start=_dt("2026-01-01T00:00:00"), end=_dt("2026-03-31T23:59:59"))
    test_range = DateRange(start=_dt("2026-04-01T00:00:00"), end=_dt("2026-04-20T23:59:59"))

    train_fraud_mix = {
        "card_testing_then_big_purchase": 0.36,
        "account_takeover_transfer": 0.30,
        "invoice_manipulation": 0.22,
        "gift_cards_social_engineering": 0.12,
    }
    test_fraud_mix = {
        "card_testing_then_big_purchase": 0.26,
        "account_takeover_transfer": 0.24,
        "invoice_manipulation": 0.20,
        "gift_cards_social_engineering": 0.12,
        "crypto_topup_drift": 0.18,  # concept drift pattern
    }

    train_users, train_txns, train_msgs = _build_split(
        seed=args.seed,
        n_users=220,
        dr=train_range,
        fraud_mix=train_fraud_mix,
        hard_mode=False,
    )
    test_users, test_txns, test_msgs = _build_split(
        seed=args.seed + 7,
        n_users=90,
        dr=test_range,
        fraud_mix=test_fraud_mix,
        hard_mode=True,
    )

    write_split(os.path.join(out_root, "train"), train_users, train_txns, train_msgs)
    write_split(os.path.join(out_root, "test"), test_users, test_txns, test_msgs)

    # Optional "public" test files (no labels) plus label-only files for facilitators.
    test_public = test_txns.drop(columns=[c for c in ["is_fraud", "fraud_type", "fraud_loss_eur"] if c in test_txns.columns]).copy()
    test_labels = test_txns[["transaction_id", "is_fraud", "fraud_type"]].copy()
    test_public.to_csv(os.path.join(out_root, "test", "transactions_public.csv"), index=False)
    test_labels.to_csv(os.path.join(out_root, "test", "transactions_labels.csv"), index=False)

    msg_public = test_msgs.drop(columns=[c for c in ["is_phishing"] if c in test_msgs.columns]).copy()
    msg_labels = test_msgs[["message_id", "is_phishing"]].copy()
    msg_public.to_csv(os.path.join(out_root, "test", "messages_public.csv"), index=False)
    msg_labels.to_csv(os.path.join(out_root, "test", "messages_labels.csv"), index=False)

    summary = {
        "train": {
            "users": int(train_users.shape[0]),
            "transactions": int(train_txns.shape[0]),
            "fraud_txns": int(train_txns["is_fraud"].sum()),
            "messages": int(train_msgs.shape[0]),
            "phish_messages": int(train_msgs["is_phishing"].sum()),
        },
        "test": {
            "users": int(test_users.shape[0]),
            "transactions": int(test_txns.shape[0]),
            "fraud_txns": int(test_txns["is_fraud"].sum()),
            "messages": int(test_msgs.shape[0]),
            "phish_messages": int(test_msgs["is_phishing"].sum()),
        },
    }
    pd.DataFrame(summary).to_json(os.path.join(out_root, "dataset_summary.json"), indent=2)
    print(pd.DataFrame(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
