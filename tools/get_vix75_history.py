import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os

# ============================================================
# CONFIG
# ============================================================

SYMBOL = "Volatility 75 Index"               # Make sure your broker uses this exact name
TIMEFRAMES = {
    #"M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H4": mt5.TIMEFRAME_H4,
    #"D1": mt5.TIMEFRAME_D1
}

# How far back to download
YEARS_BACK = 1.5

# Output folder
OUTPUT_DIR = "./historical_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def init_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Failed to select symbol {SYMBOL}. Check your MT5 market watch name.")

    print(f"[OK] Connected to MT5 and selected symbol: {SYMBOL}")


def download_data_simple(symbol, timeframe, start_time):
    """
    Downloads data iteratively starting from a specific time until the present.
    This avoids complex manual chunk calculations and guarantees the latest data.
    """
    result = []
    
    # Request data starting from the specified time
    # We rely on MT5 to fetch data up to the present moment if available
    rates = mt5.copy_rates_range(symbol, timeframe, start_time, datetime.now(pytz.utc))

    if rates is None:
        raise RuntimeError(f"Failed to copy rates: {mt5.last_error()}")
    
    if len(rates) == 0:
        raise RuntimeError("No data downloaded. Check your date range or symbol name.")
    else:
        result.append(pd.DataFrame(rates))
        print(f"[INFO] Downloaded {len(rates)} bars from {start_time.strftime('%Y-%m-%d %H:%M')}")

    df = pd.concat(result, ignore_index=True)
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time")

    return df

def process_dataframe(df):
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open": "Open",
        "spread": "Spread",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume"
    })
    df = df[["Open", "High", "Low", "Close", "Volume", "Spread"]]

    df = df[df.index.notnull()]
    df = df[df["Open"].notnull()]

    return df


def save_data(df, name):
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    parquet_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")

    df.to_csv(csv_path)
    df.to_parquet(parquet_path)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {parquet_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    init_mt5()

    end = datetime.now(pytz.utc)
    start = end - timedelta(days=YEARS_BACK * 365)

    for label, tf in TIMEFRAMES.items():
        print(f"\n=== Downloading {SYMBOL} {label} ===")

        raw_df = download_data_simple(SYMBOL, tf, start)
        print(f"[INFO] Raw rows: {len(raw_df)}")

        df = process_dataframe(raw_df)
        print(f"[INFO] Clean rows: {len(df)}")

        save_data(df, f"{SYMBOL}_{label}")

    mt5.shutdown()
    print("\n[COMPLETE] All data downloaded and stored.")


if __name__ == "__main__":
    main()
 
