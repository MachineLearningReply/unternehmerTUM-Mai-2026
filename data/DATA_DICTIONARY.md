# Data Dictionary (Fraud Detection Workshop)

The dataset is split into `train/` and `test/`. Each split contains:

- `users.csv`: one row per customer
- `transactions.csv`: one row per bank transaction (labeled with `is_fraud`)
- `messages.csv`: one row per text message / email received by the customer (labeled with `is_phishing`)

All timestamps are stored as ISO-8601 strings with an explicit UTC offset (example: `2026-01-28T09:36:00+01:00`).

### Public vs labeled test files

In `data/test/` you also have:
- `transactions_public.csv`: same schema as `transactions.csv` but without `is_fraud` / `fraud_type`
- `transactions_labels.csv`: `transaction_id` + labels (for facilitators)
- `messages_public.csv`: messages without `is_phishing`
- `messages_labels.csv`: `message_id` + label (for facilitators)

## `users.csv`

| Column | Type | Meaning |
|---|---:|---|
| `user_id` | string | Customer identifier (stable across tables) |
| `age` | int | Customer age (years) |
| `employment_status` | string | `student`, `employed`, `self_employed`, `unemployed` |
| `segment` | string | Coarse business segment (e.g., `Student`, `Mass`, `Affluent`) |
| `home_country` | string | ISO-like 2-letter country code |
| `home_city` | string | Home city |
| `account_open_date` | date | Account opening date (YYYY-MM-DD) |
| `account_age_days` | int | Tenure in days at the end of the split |
| `has_2fa` | int | Whether 2-factor auth is enabled (0/1) |
| `device_count` | int | Number of known devices (approximate) |
| `credit_score` | int | Synthetic credit score (roughly 420–850) |
| `risk_profile` | string | `low`, `medium`, `high` (customer risk appetite) |
| `kyc_tier` | string | `basic`, `standard`, `enhanced` |
| `monthly_income_eur` | float | Approx. monthly income in EUR |
| `avg_monthly_spend_eur` | float | Approx. monthly spend in EUR |

## `transactions.csv`

| Column | Type | Meaning |
|---|---:|---|
| `transaction_id` | string | Transaction identifier |
| `user_id` | string | Customer identifier |
| `timestamp` | string | Transaction timestamp |
| `direction` | string | `debit` (outgoing) or `credit` (incoming) |
| `amount_eur` | float | Amount in EUR (already normalized) |
| `currency` | string | Currency code (mostly `EUR`) |
| `transaction_type` | string | `CARD`, `TRANSFER`, `ATM`, `DIRECT_DEBIT`, `INCOMING` |
| `merchant_name` | string | Merchant / counterparty display name |
| `merchant_category` | string | Category (e.g., `groceries`, `rent`, `electronics`, `gift_cards`, `crypto`) |
| `merchant_country` | string | Merchant / counterparty country |
| `channel` | string | `card`, `bank_transfer`, `direct_debit`, `atm` |
| `entry_mode` | string | For cards: `chip`, `contactless`, `online`; else `not_applicable` |
| `device_id` | string | Device used (synthetic) |
| `ip_country` | string | Country derived from session IP (synthetic) |
| `is_international` | int | 1 if `merchant_country != home_country` else 0 |
| `is_recurring` | int | 1 if part of a recurring pattern else 0 |
| `recurring_name` | string | Recurring label (e.g., `Rent`, `Streaming`) or `none` |
| `beneficiary_id` | string | For transfers: synthetic beneficiary id; else `none` |
| `is_fraud` | int | Ground-truth label (0/1) |
| `fraud_type` | string | Fraud scenario label (`none` if not fraud) |
| `hour` | int | Hour-of-day derived from `timestamp` |
| `day_of_week` | int | Monday=0 … Sunday=6 |
| `is_weekend` | int | Weekend flag (0/1) |
| `merchant_seen_before` | int | 1 if this user has transacted with this merchant earlier in the split |
| `device_seen_before` | int | 1 if this user has used this device earlier in the split |
| `potential_loss_eur` | float | Potential loss proxy (equals amount for debits; 0 for credits) |
| `review_cost_eur` | float | Operational review cost proxy (fixed) |
| `fraud_loss_eur` | float | Realized fraud loss proxy (`amount_eur` for fraud debits; else 0) |

Notes:
- `merchant_seen_before` and `device_seen_before` are computed **within each split** (no train→test leakage).
- Some transactions are designed to be “easy” fraud (high amount, new device, unusual IP), and some are intentionally subtle (looks like a normal bill, but has correlated phishing messages and atypical context).

## `messages.csv`

| Column | Type | Meaning |
|---|---:|---|
| `message_id` | string | Message identifier |
| `user_id` | string | Customer identifier |
| `timestamp` | string | Message timestamp |
| `channel` | string | `SMS` or `Email` |
| `sender` | string | Sender display name / email-like string |
| `message_text` | string | Message content |
| `is_phishing` | int | Ground-truth phishing label (0/1) |
| `hour` | int | Hour-of-day derived from `timestamp` |
| `day_of_week` | int | Monday=0 … Sunday=6 |
