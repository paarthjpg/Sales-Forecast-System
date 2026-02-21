import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# ── Date range
start_date = datetime(2021, 1, 1)
end_date   = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

products = ['Electronics', 'Clothing', 'Food', 'Furniture']
regions  = ['North', 'South', 'East', 'West']

# ── Product multipliers (Electronics sells more; Food is lower-ticket)
product_base = {
    'Electronics': 1300,
    'Furniture':   1100,
    'Clothing':     950,
    'Food':         850,
}

# ── Region multipliers (North & West are wealthier markets)
region_mult = {
    'North': 1.12,
    'West':  1.08,
    'East':  1.00,
    'South': 0.93,
}

# ── Product-specific seasonality (Electronics peaks in Dec; Food peaks in summer)
def product_seasonal(product, month):
    if product == 'Electronics':
        return 250 * np.sin(2 * np.pi * (month - 11) / 12)   # peaks Dec
    elif product == 'Food':
        return 180 * np.sin(2 * np.pi * (month - 6) / 12)    # peaks Jul
    elif product == 'Clothing':
        return 150 * np.sin(2 * np.pi * (month - 4) / 12)    # peaks Apr/Oct
    elif product == 'Furniture':
        return 120 * np.sin(2 * np.pi * (month - 5) / 12)    # peaks May
    return 0

# ── Product-specific weekend boost
weekend_boost = {
    'Electronics': 200,
    'Clothing':     170,
    'Furniture':    130,
    'Food':          80,
}

# ── Product-specific noise level
noise_std = {
    'Electronics': 80,
    'Furniture':   75,
    'Clothing':    70,
    'Food':        60,
}

rows = []
total_days = len(dates)

for i, date in enumerate(dates):
    trend = i * 0.55                        # ~$0.55/day across 3 years ≈ +$600 total growth
    month    = date.month
    is_wkend = date.weekday() >= 5

    for product in products:
        for region in regions:
            base      = product_base[product]
            seasonal  = product_seasonal(product, month)
            wkend     = weekend_boost[product] if is_wkend else 0
            region_sc = region_mult[region]
            noise     = np.random.normal(0, noise_std[product])
            sales     = (base + trend + seasonal + wkend + noise) * region_sc
            rows.append({
                'Date':    date,
                'Product': product,
                'Region':  region,
                'Sales':   round(max(0, sales), 2),
            })

df = pd.DataFrame(rows)
df.to_csv('data/raw/sales_data.csv', index=False)

print(f"Created dataset: {len(df):,} rows  ({len(dates)} days × {len(products)} products × {len(regions)} regions)")
print(df.groupby(['Product','Region'])['Sales'].mean().unstack().round(0))
print(f"\nDate range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Total sales: ${df['Sales'].sum():,.0f}")