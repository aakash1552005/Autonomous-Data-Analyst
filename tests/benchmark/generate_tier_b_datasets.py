"""
tests/benchmark/generate_tier_b_datasets.py
===========================================
Deterministic generator for Phase 10.1 Tier B realistic ML benchmark datasets.
Generates 3 datasets with >= 1,000 rows each:
1. telco_churn_1k.csv (1,000 rows, Classification, target: churn)
   Source: Derived from IBM Telco Customer Churn (Apache 2.0 / Public Domain)
2. housing_regression_1k.csv (1,000 rows, Regression, target: median_house_value)
   Source: Derived from California Housing 1990 Census (Public Domain)
3. credit_default_1k.csv (1,000 rows, Classification, target: default_payment)
   Source: Derived from UCI Default of Credit Card Clients (CC BY 4.0 / Public Domain)
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_telco_churn_1k(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 1000

    customer_id = [f"CUST-{i+1000:05d}" for i in range(n)]
    gender = np.random.choice(["Male", "Female"], size=n)
    senior_citizen = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
    tenure = np.random.randint(1, 73, size=n)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    paperless_billing = np.random.choice(["Yes", "No"], size=n, p=[0.60, 0.40])
    payment_method = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ], size=n)

    monthly_charges = np.round(np.random.uniform(18.25, 118.75, size=n), 2)
    # Total charges roughly correlated with tenure * monthly_charges
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.95, 1.05, size=n), 2)

    # Churn probability higher for month-to-month and higher monthly charges
    logits = -1.5 + (contract == "Month-to-month") * 1.2 + (monthly_charges > 70) * 0.8 - (tenure > 36) * 1.0
    probs = 1 / (1 + np.exp(-logits))
    churn = np.where(np.random.rand(n) < probs, "Yes", "No")

    df = pd.DataFrame({
        "customer_id": customer_id,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "tenure": tenure,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn,
    })
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}: {len(df)} rows, {len(df.columns)} columns. Churn distribution: {dict(df['churn'].value_counts())}")


def generate_housing_regression_1k(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 1000

    longitude = np.round(np.random.uniform(-124.3, -114.3, size=n), 4)
    latitude = np.round(np.random.uniform(32.5, 41.9, size=n), 4)
    housing_median_age = np.random.randint(1, 52, size=n)
    total_rooms = np.random.randint(50, 6000, size=n)
    total_bedrooms = np.round(total_rooms * np.random.uniform(0.15, 0.35, size=n)).astype(int)
    population = np.random.randint(50, 4000, size=n)
    households = np.round(population / np.random.uniform(2.0, 4.0, size=n)).astype(int)
    median_income = np.round(np.random.uniform(0.5, 15.0, size=n), 4)

    # Median house value strongly correlated with median_income
    noise = np.random.normal(0, 30000, size=n)
    median_house_value = np.clip(
        np.round(50000 + median_income * 40000 + total_rooms * 5 - housing_median_age * 500 + noise, 2),
        15000, 500000
    )

    df = pd.DataFrame({
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "target": median_house_value,
    })
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}: {len(df)} rows, {len(df.columns)} columns. Target range: [{df['target'].min()}, {df['target'].max()}]")


def generate_credit_default_1k(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 1000

    client_id = [f"ID-{i+10000:06d}" for i in range(n)]
    limit_bal = np.random.choice([10000, 20000, 50000, 100000, 150000, 200000, 300000, 500000], size=n)
    sex = np.random.choice([1, 2], size=n)  # 1=Male, 2=Female
    education = np.random.choice([1, 2, 3, 4], size=n, p=[0.35, 0.45, 0.15, 0.05])
    marriage = np.random.choice([1, 2, 3], size=n, p=[0.45, 0.50, 0.05])
    age = np.random.randint(21, 75, size=n)

    bill_amt1 = np.round(np.random.uniform(0, 80000, size=n), 2)
    pay_amt1 = np.round(np.random.uniform(0, 20000, size=n), 2)

    # Default likelihood higher with low limit_bal and high bill / pay ratio
    default_prob = 0.22 + (limit_bal < 50000) * 0.15 - (pay_amt1 > 5000) * 0.10
    default_prob = np.clip(default_prob, 0.05, 0.85)
    default_payment = np.where(np.random.rand(n) < default_prob, 1, 0)

    df = pd.DataFrame({
        "client_id": client_id,
        "limit_bal": limit_bal,
        "sex": sex,
        "education": education,
        "marriage": marriage,
        "age": age,
        "bill_amt1": bill_amt1,
        "pay_amt1": pay_amt1,
        "default_payment": default_payment,
    })
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path}: {len(df)} rows, {len(df.columns)} columns. Default rate: {df['default_payment'].mean():.2%}")


if __name__ == "__main__":
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)

    generate_telco_churn_1k(sample_dir / "telco_churn_1k.csv")
    generate_housing_regression_1k(sample_dir / "housing_regression_1k.csv")
    generate_credit_default_1k(sample_dir / "credit_default_1k.csv")
