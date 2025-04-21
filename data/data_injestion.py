import time
from datetime import datetime, timedelta

import pandas as pd
import requests


def fetch_histohour_chunk(to_timestamp, limit=2000):
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {"fsym": "BTC", "tsym": "USD", "limit": limit - 1, "toTs": to_timestamp}

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["Response"] == "Success":
            return pd.DataFrame(data["Data"]["Data"])
        else:
            print("API error:", data["Message"])
            return pd.DataFrame()
    else:
        print("Failed:", response.status_code)
        return pd.DataFrame()


# Pull in reverse chunks from now back to 1 year ago
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=4000)

all_chunks = []
current_end = int(end_time.timestamp())

while True:
    chunk = fetch_histohour_chunk(current_end)
    if chunk.empty:
        break

    # Drop duplicate last row if overlapping with previous chunk
    if all_chunks:
        chunk = chunk.iloc[:-1]

    all_chunks.append(chunk)

    # Update time window for next chunk
    earliest_time = chunk["time"].min()
    if datetime.utcfromtimestamp(earliest_time) < start_time:
        break

    current_end = earliest_time
    time.sleep(1)

# Combine and process
df = pd.concat(all_chunks, ignore_index=True)
df["timestamp"] = pd.to_datetime(df["time"], unit="s")
df = df[["timestamp", "open", "high", "low", "close", "volumefrom", "volumeto"]]
df.rename(columns={"volumefrom": "volume_btc", "volumeto": "volume_usd"}, inplace=True)
df.drop_duplicates(subset="timestamp", inplace=True)
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# Save to CSV
df.to_csv("btc_hourly_ohlc_volume_1year_cryptocompare.csv", index=False)
print("✅ Saved: btc_hourly_ohlc_volume_1year_cryptocompare.csv")
