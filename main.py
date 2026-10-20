# imports
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# download and load csv
url = "https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv"
try:
    df = pd.read_csv(url)
    df.to_csv("marketing.csv", index=False)
    print("fetched and saved marketing.csv")
except Exception as e:
    print("Eroor: ", e)


def read_clean_data():
    df = pd.read_csv("marketing.csv")
    df.drop(columns=['month', 'year'], inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce')
    return df

df = read_clean_data()

# Shape and Structural Analysis
print("DATASET SHAPE & STRUCTURE")
print(f"Shape (Rows, Cols) : {df.shape}\n")

print("Column Data Types:")
print(df.dtypes.to_string())

print("\nMissing Values Per Column:")
print(df.isna().sum().to_string())

display(df.head())
display(df.describe().round(1))

# Sort by date for proper time-series plotting
df_sorted = df.sort_values('date').dropna(subset=['date'])

fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(df_sorted['date'], df_sorted['priceMax'], color='crimson', alpha=0.6, label='Max Price')
ax1.plot(df_sorted['date'], df_sorted['priceMin'], color='royalblue', alpha=0.6, label='Min Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='black')
ax1.set_title('Market Price Trends Over Time')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# NumPy Statistical Operations
max_prices = df['priceMax'].to_numpy()
min_prices = df['priceMin'].to_numpy()

print(f"The highest max price : {max_prices.max()} on date: {df['date'].iloc[max_prices.argmax()].date()}")
print(f"The lowest max price  : {max_prices.min()} on date: {df['date'].iloc[max_prices.argmin()].date()}")
print(f"The average max price : {max_prices.mean():.1f}")

low_price_days = (min_prices < 100).sum()
print(f"Days with very low min prices (< 100): {int(low_price_days)} out of {len(min_prices)} days.")