import argparse
import os
import sys
import requests
import pandas as pd

DATA_DIR = r"C:\Projects\Kronos\data"

OKX_BAR = {"1d": "1D", "1h": "1H", "5m": "5m", "15m": "15m", "4h": "4H"}


def fetch_crypto_binance(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=20)
    r.raise_for_status()
    rows = []
    for k in r.json():
        rows.append({
            "timestamps": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "amount": float(k[7]),
        })
    return pd.DataFrame(rows)


def fetch_crypto_okx(symbol, interval, limit):
    inst = symbol.upper().replace("USDT", "-USDT")
    bar = OKX_BAR.get(interval, interval)
    url = "https://www.okx.com/api/v5/market/candles"
    r = requests.get(url, params={"instId": inst, "bar": bar, "limit": limit}, timeout=20)
    r.raise_for_status()
    data = r.json().get("data", [])
    rows = []
    for k in reversed(data):
        vol = float(k[5])
        close = float(k[4])
        rows.append({
            "timestamps": pd.to_datetime(int(k[0]), unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": close,
            "volume": vol,
            "amount": vol * close,
        })
    return pd.DataFrame(rows)


def fetch_crypto(symbol, interval, limit):
    try:
        df = fetch_crypto_binance(symbol, interval, limit)
        print(f"[crypto] Binance OK: {len(df)} velas")
        return df
    except Exception as e:
        print(f"[crypto] Binance fallo ({str(e)[:80]}), probando OKX...")
        df = fetch_crypto_okx(symbol, interval, limit)
        print(f"[crypto] OKX OK: {len(df)} velas")
        return df


def fetch_us(symbol, interval, period):
    import yfinance as yf
    df = yf.download(symbol.upper(), period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance no devolvio datos para {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    out = pd.DataFrame({
        "timestamps": pd.to_datetime(df[ts_col]),
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
        "volume": df["Volume"].astype(float),
        "amount": (df["Close"].astype(float) * df["Volume"].astype(float)),
    })
    print(f"[us] yfinance OK: {len(out)} velas")
    return out


def main():
    p = argparse.ArgumentParser(description="Descarga datos frescos para el Kronos Web UI")
    p.add_argument("source", choices=["crypto", "us"])
    p.add_argument("symbol", help="p.ej. BTCUSDT o AAPL")
    p.add_argument("--interval", default="1d", help="1d (def), 1h, 5m ...")
    p.add_argument("--limit", type=int, default=1000, help="velas (solo crypto)")
    p.add_argument("--period", default="2y", help="periodo (solo us), p.ej. 2y, 5y")
    p.add_argument("--outdir", default=DATA_DIR)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.source == "crypto":
        df = fetch_crypto(args.symbol, args.interval, args.limit)
    else:
        df = fetch_us(args.symbol, args.interval, args.period)

    df = df.dropna().reset_index(drop=True)
    fname = f"{args.symbol.upper()}_{args.source}_{args.interval}.csv"
    fpath = os.path.join(args.outdir, fname)
    df.to_csv(fpath, index=False)

    print(f"Guardado: {fpath}")
    print(f"Rango: {df['timestamps'].iloc[0]}  ->  {df['timestamps'].iloc[-1]}")
    print(f"Filas: {len(df)} (la UI necesita >= 520)")


if __name__ == "__main__":
    sys.exit(main())
