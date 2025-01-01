import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1000,
    n_rows: int = 5000
):
    np.random.seed(9999)

    dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()

    df = pd.DataFrame(
        {
            "date": dates,
            "average_temperature": np.random.uniform(10, 35, n_rows),
            "rainfall": np.random.exponential(5, n_rows),
            "weekend": [(date.weekday() >= 5) * 1 for date in dates],
            "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
            "price_per_kg": np.random.uniform(0.5, 3, n_rows),
            "month": [date.month for date in dates]
        }
    )

    df["inflation_multiplier"] = (
        1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03
    )

    df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3)/12) + np.sin(2 * np.pi * df["month"] - 9/12)

    df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5

    peak_months = [4, 10]
    df["promo"] = np.where(
        df["month"].isin(peak_months),
        1,
        np.random.choice([0, 1], n_rows, p=[0.85, 0.15])
    )

    base_price_effect = -df["price_per_kg"] * 50
    seasonary_effect = df["harvest_effect"] * 50
    promo_effect = df["promo"] * 200

    df["demand"] = (
        base_demand
        + base_price_effect
        + seasonary_effect
        + promo_effect
        + df["weekend"] * 300
        + np.random.normal(0, 50, n_rows)
    ) * df["inflation_multiplier"]

    df["previous_days_demand"] = df["demand"].shift(1)
    df["previous_days_demand"].fillna(method="bfill", inplace=True)

    df.drop(columns=["inflation_multiplier", "harvest_effect", "month"], inplace=True)

    return df

data = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=1_000)

print(data[-20:])