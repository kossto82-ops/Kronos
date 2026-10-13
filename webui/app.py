import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import time
import requests
import threading
import hashlib
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

@app.after_request
def add_no_cache(response):
    """Prevent browsers from caching the HTML so UI changes show immediately"""
    if 'text/html' in response.headers.get('Content-Type', ''):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Global variables to store models
tokenizer = None
model = None
predictor = None
CURRENT_MODEL_KEY = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Normalize column names to lowercase (CSVs often use Date/Open/...)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Missing required columns: {missing} (found: {list(df.columns)})"
        
        # Process timestamp column
        ts_col = next((c for c in ['timestamps', 'timestamp', 'date', 'datetime', 'time'] if c in df.columns), None)
        if ts_col is not None:
            df['timestamps'] = pd.to_datetime(df[ts_col], errors='coerce')
            if df['timestamps'].isna().any():
                df = df.dropna(subset=['timestamps'])
        else:
            # No date column: generate synthetic hourly timestamps (flagged for the UI)
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')
            df.attrs['synthetic_timestamps'] = True
        
        # Ensure numeric columns are numeric type
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume/amount columns (optional)
        for col in ['volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by time (stable) and keep only the last value per timestamp
        df = df.sort_values('timestamps', kind='stable')
        df = df.drop_duplicates(subset=['timestamps'], keep='last')
        
        # Remove rows with invalid values only in the columns used for prediction
        subset = required_cols + (['volume'] if 'volume' in df.columns else [])
        before = len(df)
        df = df.dropna(subset=subset)
        if before != len(df):
            df.attrs['dropped_rows'] = before - len(df)
        
        if len(df) == 0:
            return None, "File contains no valid rows after cleaning"
        
        df = df.reset_index(drop=True)
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

OKX_BAR = {"1d": "1D", "1h": "1H", "5m": "5m", "15m": "15m", "4h": "4H"}
US_PERIOD_BY_INTERVAL = {"1d": "3y", "1h": "730d", "5m": "60d", "15m": "60d", "30m": "60d"}

PRESET_TICKERS = {
    'ai': [
        'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AAPL', 'ORCL', 'PLTR',
        'CRM', 'SNOW', 'IBM', 'NOW', 'AMD', 'AVGO', 'MRVL'
    ],
    'chips': [
        'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT',
        'LRCX', 'ASML', 'ADI', 'NXPI', 'MRVL', 'ON', 'KLAC', 'ARM', 'SMCI'
    ],
    'quantum': [
        'IONQ', 'RGTI', 'QBTS', 'QUBT', 'ARQQ', 'IBM', 'GOOGL', 'MSFT', 'HON'
    ],
}


def _fetch_crypto_binance(symbol, interval, limit):
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        timeout=20,
    )
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


def _fetch_crypto_okx(symbol, interval, limit):
    inst = symbol.upper().replace("USDT", "-USDT")
    r = requests.get(
        "https://www.okx.com/api/v5/market/candles",
        params={"instId": inst, "bar": OKX_BAR.get(interval, interval), "limit": limit},
        timeout=20,
    )
    r.raise_for_status()
    rows = []
    for k in reversed(r.json().get("data", [])):
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


def fetch_live_data(source, symbol, interval="1d", limit=1000, period=None):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol cannot be empty")

    if source == "crypto":
        try:
            return _fetch_crypto_binance(symbol, interval, limit)
        except Exception:
            return _fetch_crypto_okx(symbol, interval, limit)

    if source == "us":
        import yfinance as yf
        period = period or US_PERIOD_BY_INTERVAL.get(interval, "2y")
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no data for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
        return pd.DataFrame({
            "timestamps": pd.to_datetime(df[ts_col]),
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
            "volume": df["Volume"].astype(float),
            "amount": df["Close"].astype(float) * df["Volume"].astype(float),
        })

    raise ValueError(f"Unsupported source: {source}")


def preset_union(name):
    """Return tickers for a sector preset, or the union of all sectors for 'all'"""
    if name == 'all':
        seen = []
        for key in ('ai', 'chips', 'quantum'):
            for ticker in PRESET_TICKERS[key]:
                if ticker not in seen:
                    seen.append(ticker)
        return seen
    return list(PRESET_TICKERS.get(name, []))


@app.route('/api/fetch-data', methods=['POST'])
def fetch_data():
    """Download fresh market data and save it into the data directory"""
    try:
        payload = request.get_json() or {}
        source = payload.get('source', 'crypto')
        symbol = payload.get('symbol', '')
        interval = payload.get('interval', '1d')
        limit = int(payload.get('limit', 1000))
        preset = payload.get('preset') or payload.get('sector')

        if source == 'us' and preset:
            tickers = preset_union(preset)
            if not tickers:
                return jsonify({'error': f'Unknown preset: {preset}'}), 400
            files = []
            errors = []
            for ticker in tickers:
                try:
                    df = fetch_live_data(source, ticker, interval=interval)
                    df = df.dropna().reset_index(drop=True)
                    if df.empty:
                        errors.append(f"{ticker}: no data")
                        continue
                    os.makedirs(DATA_DIR, exist_ok=True)
                    file_name = f"{ticker}_{source}_{interval}.csv"
                    df.to_csv(os.path.join(DATA_DIR, file_name), index=False)
                    warning = None
                    if len(df) < 520:
                        warning = f"Only {len(df)} rows (< 520 needed for a full prediction window)"
                    files.append({
                        'file_name': file_name,
                        'file_path': os.path.join(DATA_DIR, file_name),
                        'rows': len(df),
                        'start': str(df['timestamps'].iloc[0]),
                        'end': str(df['timestamps'].iloc[-1]),
                        'warning': warning,
                    })
                    print(f"[fetch-data] Saved {file_name} ({len(df)} rows)")
                except Exception as e:
                    errors.append(f"{ticker}: {str(e)}")
                    print(f"[fetch-data] {ticker} failed: {e}")

            if not files:
                return jsonify({'error': f'All {len(tickers)} tickers failed: ' + '; '.join(errors[:5])}), 502

            detail = f"Downloaded {len(files)}/{len(tickers)} files for preset '{preset}'"
            if errors:
                detail += f" ({len(errors)} failed: " + '; '.join(errors[:5]) + ')'
            return jsonify({
                'success': True,
                'message': detail,
                'files': files,
                'errors': errors,
            })

        if not symbol:
            return jsonify({'error': 'Enter a symbol (or select a sector preset)'}), 400

        df = fetch_live_data(source, symbol, interval=interval, limit=limit)
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            return jsonify({'error': 'No data returned for the requested symbol'}), 400

        os.makedirs(DATA_DIR, exist_ok=True)
        file_name = f"{symbol.strip().upper()}_{source}_{interval}.csv"
        file_path = os.path.join(DATA_DIR, file_name)
        df.to_csv(file_path, index=False)
        warning = None
        if len(df) < 520:
            warning = f"Only {len(df)} rows (< 520 needed for a full prediction window)"

        return jsonify({
            'success': True,
            'file_name': file_name,
            'file_path': file_path,
            'rows': len(df),
            'start': str(df['timestamps'].iloc[0]),
            'end': str(df['timestamps'].iloc[-1]),
            'warning': warning,
            'files': [{
                'file_name': file_name,
                'file_path': file_path,
                'rows': len(df),
                'start': str(df['timestamps'].iloc[0]),
                'end': str(df['timestamps'].iloc[-1]),
                'warning': warning,
            }],
            'errors': [],
            'message': f'Downloaded {len(df)} candles for {symbol.upper()} ({interval}) -> {file_name}'
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error while downloading data: {str(e)}'}), 502
    except Exception as e:
        return jsonify({'error': f'Failed to download data: {str(e)}'}), 500


def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # Continuity analysis only makes sense with both prediction and actual data
        if prediction_results and actual_data:
            last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point

            save_data['analysis']['continuity'] = {
                'last_prediction': {
                    'open': last_pred['open'],
                    'high': last_pred['high'],
                    'low': last_pred['low'],
                    'close': last_pred['close']
                },
                'first_actual': {
                    'open': first_actual['open'],
                    'high': first_actual['high'],
                    'low': first_actual['low'],
                    'close': first_actual['close']
                },
                'gaps': {
                    'open_gap': abs(last_pred['open'] - first_actual['open']),
                    'high_gap': abs(last_pred['high'] - first_actual['high']),
                    'low_gap': abs(last_pred['low'] - first_actual['low']),
                    'close_gap': abs(last_pred['close'] - first_actual['close'])
                },
                'gap_percentages': {
                    'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                    'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                    'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                    'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                }
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_overview_chart(df, max_points=600):
    """Create a candlestick preview chart for the loaded data"""
    view = df.tail(max_points).reset_index(drop=True)
    x = view['timestamps'] if 'timestamps' in view.columns else view.index
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x,
        open=view['open'],
        high=view['high'],
        low=view['low'],
        close=view['close'],
        name='OHLC',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    fig.update_layout(
        title=f'Loaded data preview - last {len(view)} of {len(df)} candles',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        warnings = []
        if df.attrs.get('synthetic_timestamps'):
            warnings.append('The file has no date column: synthetic hourly timestamps were generated, dates shown are NOT real')
        if df.attrs.get('dropped_rows'):
            warnings.append(f"Removed {df.attrs['dropped_rows']} rows with invalid values")

        last_date = None
        days_old = None
        if 'timestamps' in df.columns and not df.attrs.get('synthetic_timestamps'):
            last_date = pd.Timestamp(df['timestamps'].max())
            days_old = max(0, (pd.Timestamp.now().normalize() - last_date.normalize()).days)

        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'last_date': last_date.isoformat() if last_date is not None else None,
            'days_old': days_old,
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'chart': create_overview_chart(df),
            'warnings': warnings,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

def build_prediction_summary(prediction_results, x_df, pred_len, actual_data=None, is_forward=False):
    """Build a plain-language summary about where the prediction goes (up/down/sideways).

    is_forward=True means the prediction covers candles BEYOND the last real data point;
    otherwise it is a backtest over a window that already exists in the file.
    """
    from_price = float(x_df['close'].iloc[-1]) if x_df is not None and len(x_df) > 0 else 0.0
    pred_closes = [float(r['close']) for r in prediction_results]
    pred_times = [r['timestamp'] for r in prediction_results]

    if not pred_closes:
        return {'direction': 'unknown', 'explanation': 'No hay datos de predicción disponibles'}

    window = min(5, len(pred_closes))
    end_price = float(np.mean(pred_closes[-window:]))
    change_pct = (end_price - from_price) / from_price * 100.0 if from_price else 0.0

    threshold = 0.15
    if change_pct > threshold:
        direction = 'up'
        direction_label = 'SUBIR'
    elif change_pct < -threshold:
        direction = 'down'
        direction_label = 'BAJAR'
    else:
        direction = 'sideways'
        direction_label = 'LATERAL'

    lo = float(min(pred_closes))
    hi = float(max(pred_closes))
    range_pct = (hi - lo) / from_price * 100.0 if from_price else 0.0

    pct_series = []
    prev = from_price
    for c in pred_closes:
        if prev:
            pct_series.append((c - prev) / prev * 100.0)
        prev = c
    daily_vol = float(np.mean([abs(p) for p in pct_series])) if pct_series else 0.0

    fmt = lambda v: f"{v:,.2f}"
    start_date = str(pred_times[0])[:10]
    end_date = str(pred_times[-1])[:10]

    if is_forward:
        explanation = (
            f"Previsión hacia el futuro, del {start_date} al {end_date}: se espera que el precio "
            f"pueda {direction_label.lower()} alrededor de un {abs(change_pct):.2f}% respecto al "
            f"último dato real ({fmt(from_price)}) hasta {fmt(end_price)}."
        )
    else:
        explanation = (
            f"Esto es un backtest (prueba) del {start_date} al {end_date}: esas {pred_len} velas ya "
            f"ocurrieron en el fichero de datos. La predicción indicaba {direction_label.lower()} "
            f"un {abs(change_pct):.2f}% (de {fmt(from_price)} a {fmt(end_price)}), y abajo se "
            f"compara con lo que realmente pasó."
        )
    details = (
        f"El rango previsto va de {fmt(lo)} a {fmt(hi)} (±{range_pct:.2f}% del precio de partida), "
        f"con una variación típica de ±{daily_vol:.2f}% por vela."
    )

    comparison = None
    if actual_data and len(actual_data) > 0:
        actual_closes = [float(a['close']) for a in actual_data]
        actual_end = actual_closes[-1]
        actual_change_pct = (actual_end - from_price) / from_price * 100.0 if from_price else 0.0
        same_dir = (actual_change_pct > 0) == (change_pct > 0)
        n = min(len(pred_closes), len(actual_closes))
        mape = sum(abs(p - a) / a * 100.0 for p, a in zip(pred_closes, actual_closes)) / n if n else 0.0
        if actual_change_pct > 0:
            real_label = f"subidas de {actual_change_pct:.2f}%"
        elif actual_change_pct < 0:
            real_label = f"bajadas de {abs(actual_change_pct):.2f}%"
        else:
            real_label = "sin dirección clara"
        comparison = {
            'direction_matched': same_dir,
            'actual_change_pct': actual_change_pct,
            'mape': mape,
            'text': (
                f"En comparación con lo que realmente pasó después: {real_label}, "
                f"la dirección {'coincide' if same_dir else 'NO coincide'} con la predicción. "
                f"Error medio por vela: {mape:.2f}%."
            )
        }

    return {
        'direction': direction,
        'direction_label': direction_label,
        'from_price': from_price,
        'end_price': end_price,
        'change_pct': change_pct,
        'min_price': lo,
        'max_price': hi,
        'range_pct': range_pct,
        'daily_vol_pct': daily_vol,
        'pred_len': pred_len,
        'pred_closes': pred_closes,
        'pred_times': pred_times,
        'is_forward': is_forward,
        'start_date': start_date,
        'end_date': end_date,
        'explanation': explanation,
        'details': details,
        'comparison': comparison,
    }


# ---------------------------------------------------------------------------
# Walk-forward self-evaluation: does the model actually work on THIS file?
# Predicts the last N non-overlapping windows (T=0, deterministic) and
# compares direction hit-rate and end error against "do nothing" (last price).

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_results')
EVAL_STATE = {'status': 'idle', 'progress': 0, 'total': 0, 'started_at': None, 'file_path': None, 'result': None, 'error': None}
EVAL_CACHE = {}


def _eval_cache_path(file_path, lookback, pred_len):
    try:
        st = os.stat(file_path)
        key = f"{os.path.abspath(file_path)}|{st.st_size}|{int(st.st_mtime)}|{lookback}|{pred_len}"
    except OSError:
        key = f"{os.path.abspath(file_path)}|{lookback}|{pred_len}"
    name = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(EVAL_DIR, name + '.json')


def load_eval_cache(file_path, lookback=400, pred_len=120):
    cache_key = (file_path, lookback, pred_len)
    if cache_key in EVAL_CACHE:
        return EVAL_CACHE[cache_key]
    path = _eval_cache_path(file_path, lookback, pred_len)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            EVAL_CACHE[cache_key] = result
            return result
        except Exception:
            return None
    return None


def _direction(pct):
    if pct > 0.15:
        return 1
    if pct < -0.15:
        return -1
    return 0


def _slope_direction(vals):
    if len(vals) < 10:
        return 0
    x = np.arange(len(vals), dtype=float)
    y = np.asarray(vals, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    pct = slope * len(vals) / y[-1] * 100.0 if y[-1] else 0.0
    return _direction(pct)


def _trend_signals(closes):
    mom60 = ols60 = 0
    if len(closes) >= 61 and closes[-61]:
        mom60 = _direction((closes[-1] - closes[-61]) / closes[-61] * 100.0)
    if len(closes) >= 60:
        ols60 = _slope_direction(closes[-60:])
    return mom60, ols60


def _verdict_and_text(hit_rate, mape_mean, naive_mape_mean):
    if hit_rate >= 2 / 3 and mape_mean <= naive_mape_mean:
        return 'good', ('Buena: en este fichero acierta la dirección la mayoría de veces y su error '
                        'final es mejor que quedarse quieto (no hacer nada).')
    if hit_rate >= 0.5:
        return 'medium', ('Media: resultados mixtos; usa la predicción solo como señal secundaria, '
                          'nunca como única razón para operar.')
    return 'bad', ('Baja: históricamente peor que no hacer nada en este fichero; '
                   'no conviene fiarse de sus predicciones aquí.')


def _run_evaluation(file_path, lookback, pred_len, n_windows):
    try:
        df, error = load_data_file(file_path)
        if error:
            EVAL_STATE.update(status='error', error=error)
            return
        if 'timestamps' not in df.columns:
            EVAL_STATE.update(status='error', error='The file has no valid timestamps, cannot evaluate')
            return
        n_max = (len(df) - lookback) // pred_len
        if n_max < 1:
            EVAL_STATE.update(status='error', error=(
                f'Insufficient data: need at least {lookback + pred_len} rows to evaluate, file has {len(df)}'))
            return
        n = max(1, min(n_windows, n_max))
        EVAL_STATE['total'] = n
        EVAL_STATE['started_at'] = time.time()
        pred = predictor
        cols = ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else [])
        windows = []
        for i in range(n):
            end_idx = len(df) - (i + 1) * pred_len
            x_df = df.iloc[end_idx - lookback:end_idx][cols].copy()
            x_ts = df['timestamps'].iloc[end_idx - lookback:end_idx].reset_index(drop=True)
            y_ts = df['timestamps'].iloc[end_idx:end_idx + pred_len].reset_index(drop=True)
            out = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                               pred_len=pred_len, T=0.0, top_p=0.9, sample_count=1, verbose=False)
            p = out['close'].values.astype(float)
            a = df['close'].iloc[end_idx:end_idx + pred_len].values.astype(float)
            base = float(df['close'].iloc[end_idx - 1])
            closes = df['close'].iloc[end_idx - lookback:end_idx].values.astype(float)
            pred_chg = (float(p[-1]) - base) / base * 100.0
            act_chg = (float(a[-1]) - base) / base * 100.0
            pred_dir = _direction(pred_chg)
            act_dir = _direction(act_chg)
            mom60_dir, ols60_dir = _trend_signals(closes)
            mape_end = abs(float(p[-1]) - float(a[-1])) / abs(float(a[-1])) * 100.0
            naive_mape_end = abs(base - float(a[-1])) / abs(float(a[-1])) * 100.0
            windows.append({
                'start_date': str(df['timestamps'].iloc[end_idx])[:10],
                'end_date': str(df['timestamps'].iloc[end_idx + pred_len - 1])[:10],
                'pred_chg_pct': round(pred_chg, 2),
                'actual_chg_pct': round(act_chg, 2),
                'hit': int(pred_dir == act_dir),
                'mom60_hit': int(mom60_dir == act_dir),
                'ols60_hit': int(ols60_dir == act_dir),
                'inv_hit': int(-pred_dir == act_dir) if pred_dir else 0,
                'mape_end': round(mape_end, 2),
                'naive_mape_end': round(naive_mape_end, 2),
            })
            EVAL_STATE['progress'] = i + 1

        hits = sum(w['hit'] for w in windows)
        hit_rate = hits / n
        signals = {}
        for sig_key, hit_key in (('model', 'hit'), ('mom60', 'mom60_hit'),
                                 ('ols60', 'ols60_hit'), ('model_inv', 'inv_hit')):
            sig_hits = sum(w[hit_key] for w in windows)
            signals[sig_key] = {'hits': sig_hits, 'rate': round(sig_hits / n, 3)}
        best_signal = max(signals, key=lambda k: signals[k]['rate'])
        mape_mean = float(np.mean([w['mape_end'] for w in windows]))
        naive_mean = float(np.mean([w['naive_mape_end'] for w in windows]))
        verdict, verdict_text = _verdict_and_text(hit_rate, mape_mean, naive_mean)
        result = {
            'file_path': file_path,
            'file_rows': len(df),
            'lookback': lookback,
            'pred_len': pred_len,
            'n_windows': n,
            'windows': windows,
            'hits': hits,
            'hit_rate': round(hit_rate, 3),
            'signals': signals,
            'best_signal': best_signal,
            'best_signal_rate': signals[best_signal]['rate'],
            'mape_mean': round(mape_mean, 2),
            'naive_mape_mean': round(naive_mean, 2),
            'mape_better_than_naive': mape_mean <= naive_mean,
            'verdict': verdict,
            'verdict_text': verdict_text,
            'evaluated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        try:
            os.makedirs(EVAL_DIR, exist_ok=True)
            with open(_eval_cache_path(file_path, lookback, pred_len), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        EVAL_CACHE[(file_path, lookback, pred_len)] = result
        EVAL_STATE.update(status='done', result=result, error=None)
    except Exception as e:
        EVAL_STATE.update(status='error', error=str(e))


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Start a walk-forward self-evaluation for a data file (runs in background)."""
    try:
        data = request.get_json() or {}
        file_path = data.get('file_path')
        force = bool(data.get('force', False))
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        n_windows = int(data.get('n_windows', 4))

        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        if not MODEL_AVAILABLE or predictor is None:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        if not (0 < lookback <= 4096) or not (0 < pred_len <= 1024) or not (1 <= n_windows <= 12):
            return jsonify({'error': 'Invalid evaluation parameters'}), 400
        if EVAL_STATE.get('status') == 'running':
            return jsonify({'error': 'An evaluation is already running'}), 409

        cached = load_eval_cache(file_path, lookback, pred_len)
        if cached and not force:
            return jsonify({'started': False, 'cached': True, 'result': cached})

        EVAL_STATE.update(status='running', progress=0, total=0, started_at=None, file_path=file_path, result=None, error=None)
        thread = threading.Thread(target=_run_evaluation, args=(file_path, lookback, pred_len, n_windows), daemon=True)
        thread.start()
        return jsonify({'started': True, 'file_path': file_path})
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500


@app.route('/api/evaluate-status', methods=['GET'])
def evaluate_status():
    return jsonify(EVAL_STATE)


@app.route('/api/evaluate-cache', methods=['GET'])
def evaluate_cache():
    file_path = request.args.get('file_path')
    if not file_path:
        return jsonify({'error': 'file_path is required'}), 400
    try:
        lookback = int(request.args.get('lookback', 400))
        pred_len = int(request.args.get('pred_len', 120))
    except ValueError:
        return jsonify({'error': 'Invalid lookback/pred_len'}), 400
    result = load_eval_cache(file_path, lookback, pred_len)
    return jsonify({'cached': result is not None, 'result': result})


# --- Paper-trading log: record future predictions, verify against real data later ---

PAPER_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_trading_log.json')


def _load_paper_log():
    if not os.path.exists(PAPER_LOG_PATH):
        return []
    try:
        with open(PAPER_LOG_PATH, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return records if isinstance(records, list) else []
    except Exception:
        return []


def _save_paper_log(records):
    try:
        with open(PAPER_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _verify_paper_records(records):
    """Fill actual outcome for pending records whose prediction end date is now real data."""
    pending = [r for r in records if not r.get('verified') and r.get('pred_end_date')]
    by_file = {}
    for r in pending:
        by_file.setdefault(r['file_path'], []).append(r)
    changed = False
    for fp, recs in by_file.items():
        df, error = load_data_file(fp)
        if error or 'timestamps' not in df.columns:
            continue
        ts = pd.to_datetime(df['timestamps'])
        closes = df['close'].astype(float).values
        for r in recs:
            try:
                end_dt = pd.to_datetime(r['pred_end_date'])
                mask = ts >= end_dt
                if not bool(mask.any()):
                    continue
                idx = ts[mask].index[0]
                pos = ts.index.get_loc(idx)
                base = float(r.get('base_price') or 0)
                if not base:
                    continue
                actual_close = float(closes[pos])
                actual_chg = (actual_close - base) / base * 100.0
                r['verified'] = True
                r['actual_date'] = str(ts.iloc[pos])[:10]
                r['actual_close'] = round(actual_close, 4)
                r['actual_chg'] = round(actual_chg, 2)
                r['actual_dir'] = _direction(actual_chg)
                r['hit'] = int(r.get('pred_dir') is not None and r['pred_dir'] == r['actual_dir']) if r.get('pred_dir') else 0
                r['verified_at'] = datetime.datetime.now().isoformat(timespec='seconds')
                changed = True
            except Exception:
                continue
    if changed:
        _save_paper_log(records)
    return records


@app.route('/api/paper-log', methods=['GET'])
def paper_log():
    records = _verify_paper_records(_load_paper_log())
    records = sorted(records, key=lambda r: r.get('created_at') or '', reverse=True)
    verified = [r for r in records if r.get('verified')]
    hits = sum(int(r.get('hit') or 0) for r in verified)
    stats = {
        'total': len(records),
        'pending': len(records) - len(verified),
        'verified': len(verified),
        'hits': hits,
        'hit_rate': round(hits / len(verified), 3) if verified else None,
    }
    return jsonify({'records': records, 'stats': stats})


@app.route('/api/paper-log/<record_id>', methods=['DELETE'])
def paper_log_delete(record_id):
    records = [r for r in _load_paper_log() if r.get('id') != record_id]
    _save_paper_log(records)
    return jsonify({'ok': True, 'remaining': len(records)})


# --- Model comparison: walk-forward mini vs small vs base on the same windows ---

COMPARE_STATE = {'status': 'idle', 'progress': 0, 'total': 0, 'current_model': None,
                 'result': None, 'error': None}


def _compare_cache_path(file_path, lookback, pred_len):
    try:
        st = os.stat(file_path)
        key = f"{os.path.abspath(file_path)}|{st.st_size}|{int(st.st_mtime)}|{lookback}|{pred_len}|cmp"
    except OSError:
        key = f"{os.path.abspath(file_path)}|{lookback}|{pred_len}|cmp"
    name = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(EVAL_DIR, name + '.json')


def load_compare_cache(file_path, lookback, pred_len):
    path = _compare_cache_path(file_path, lookback, pred_len)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _run_compare(file_path, lookback, pred_len, n_windows):
    try:
        df, error = load_data_file(file_path)
        if error:
            COMPARE_STATE.update(status='error', error=error)
            return
        n_max = (len(df) - lookback) // pred_len
        if n_max < 1:
            COMPARE_STATE.update(status='error', error=(
                f'Insufficient data: need at least {lookback + pred_len} rows, file has {len(df)}'))
            return
        n = max(1, min(n_windows, n_max))
        cols = ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else [])

        windows = []
        mom60_hits = 0
        for i in range(n):
            end_idx = len(df) - (i + 1) * pred_len
            x_df = df.iloc[end_idx - lookback:end_idx][cols].copy()
            x_ts = df['timestamps'].iloc[end_idx - lookback:end_idx].reset_index(drop=True)
            y_ts = df['timestamps'].iloc[end_idx:end_idx + pred_len].reset_index(drop=True)
            base = float(df['close'].iloc[end_idx - 1])
            a_end = float(df['close'].iloc[end_idx + pred_len - 1])
            act_dir = _direction((a_end - base) / base * 100.0)
            closes = df['close'].iloc[end_idx - lookback:end_idx].values.astype(float)
            mom_dir, _ = _trend_signals(closes)
            mom60_hits += int(mom_dir == act_dir)
            windows.append({'x_df': x_df, 'x_ts': x_ts, 'y_ts': y_ts, 'act_dir': act_dir})

        model_keys = ['kronos-mini', 'kronos-small', 'kronos-base']
        COMPARE_STATE.update(total=len(model_keys) * n, progress=0, current_model=None)

        results_models = {}
        for mk in model_keys:
            cfg = AVAILABLE_MODELS[mk]
            COMPARE_STATE['current_model'] = cfg['name']
            tok = KronosTokenizer.from_pretrained(cfg['tokenizer_id'])
            mdl = Kronos.from_pretrained(cfg['model_id'])
            prd = KronosPredictor(mdl, tok, device='cpu', max_context=cfg['context_length'])
            hits = 0
            per_window = []
            for w in windows:
                out = prd.predict(df=w['x_df'], x_timestamp=w['x_ts'], y_timestamp=w['y_ts'],
                                  pred_len=pred_len, T=0.0, top_p=0.9, sample_count=1, verbose=False)
                p = out['close'].values.astype(float)
                base = float(w['x_df']['close'].iloc[-1])
                pred_dir = _direction((float(p[-1]) - base) / base * 100.0)
                ok = int(pred_dir == w['act_dir'])
                hits += ok
                per_window.append({'hit': ok, 'pred_dir': pred_dir, 'act_dir': w['act_dir']})
                COMPARE_STATE['progress'] += 1
            results_models[mk] = {
                'key': mk,
                'name': cfg['name'],
                'params': cfg['params'],
                'hits': hits,
                'n': n,
                'rate': round(hits / n, 3),
                'windows': per_window,
            }
            del prd, mdl, tok

        best = max(results_models, key=lambda k: results_models[k]['rate'])
        result = {
            'file_path': file_path,
            'lookback': lookback,
            'pred_len': pred_len,
            'n_windows': n,
            'models': results_models,
            'mom60': {'hits': mom60_hits, 'n': n, 'rate': round(mom60_hits / n, 3)},
            'best': best,
            'best_rate': results_models[best]['rate'],
            'recommended': best if results_models[best]['rate'] >= 0.55 else None,
            'evaluated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }
        try:
            os.makedirs(EVAL_DIR, exist_ok=True)
            with open(_compare_cache_path(file_path, lookback, pred_len), 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        COMPARE_STATE.update(status='done', result=result, error=None, current_model=None)
    except Exception as e:
        COMPARE_STATE.update(status='error', error=str(e), current_model=None)


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    try:
        data = request.get_json() or {}
        file_path = data.get('file_path')
        force = bool(data.get('force', False))
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        n_windows = int(data.get('n_windows', 4))

        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        if not (0 < lookback <= 4096) or not (0 < pred_len <= 1024) or not (1 <= n_windows <= 12):
            return jsonify({'error': 'Invalid comparison parameters'}), 400
        if COMPARE_STATE.get('status') == 'running':
            return jsonify({'error': 'A comparison is already running'}), 409

        cached = load_compare_cache(file_path, lookback, pred_len)
        if cached and not force:
            return jsonify({'started': False, 'cached': True, 'result': cached})

        COMPARE_STATE.update(status='running', progress=0, total=0, current_model=None,
                             result=None, error=None)
        thread = threading.Thread(target=_run_compare, args=(file_path, lookback, pred_len, n_windows),
                                  daemon=True)
        thread.start()
        return jsonify({'started': True, 'file_path': file_path})
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500


@app.route('/api/compare-status', methods=['GET'])
def compare_status():
    return jsonify(COMPARE_STATE)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        mode = data.get('mode', 'backtest')
        start_date = data.get('start_date')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        if mode not in ('backtest', 'future'):
            return jsonify({'error': f"Invalid mode '{mode}', expected 'backtest' or 'future'"}), 400
        if not (0 < lookback <= 4096):
            return jsonify({'error': 'lookback must be between 1 and 4096'}), 400
        if not (0 < pred_len <= 1024):
            return jsonify({'error': 'pred_len must be between 1 and 1024'}), 400
        if temperature < 0 or temperature > 5:
            return jsonify({'error': 'temperature must be between 0 (greedy/deterministic) and 5'}), 400
        if not (0 < top_p <= 1):
            return jsonify({'error': 'top_p must be in (0, 1]'}), 400
        if not (1 <= sample_count <= 32):
            return jsonify({'error': 'sample_count must be between 1 and 32'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        required_rows = max(lookback, 2) if mode == 'future' else lookback + pred_len
        if len(df) < required_rows:
            extra = '' if mode == 'future' else f' + pred_len {pred_len}'
            return jsonify({'error': f'Insufficient data: file has {len(df)} rows, need at least {required_rows} (lookback {lookback}{extra})'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Prediction mode: 'backtest' (predict a past window and compare with real data)
                # or 'future' (generate candles beyond the last real data point)
                # mode/start_date validated above

                if mode == 'future':
                    # Forward-looking prediction: use the LAST lookback real candles as input
                    # and generate pred_len NEW candles after the most recent data point.
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_input_ts = pd.date_range(
                        start=df['timestamps'].iloc[-1] + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                    y_timestamp = pd.Series(future_input_ts, name='timestamps')
                    prediction_type = (
                        f"Kronos forward prediction: last {lookback} real candles as input, "
                        f"generating {pred_len} future candles after {df['timestamps'].iloc[-1]}"
                    )
                elif start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists). Forward mode has no real
        # data to compare against, so it is skipped.
        actual_data = []
        actual_df = None
        
        if mode == 'future':
            pass  # prediction goes beyond the last real candle
        elif start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if mode == 'future':
            historical_start_idx = 0
        elif start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date and mode != 'future':
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        signal = None
        try:
            closes = x_df['close'].values.astype(float)
            base_px = float(closes[-1])
            end_px = float(prediction_results[-1]['close'])
            if base_px:
                model_pct = (end_px - base_px) / base_px * 100.0
                model_dir = _direction(model_pct)
                mom60_dir, ols60_dir = _trend_signals(closes)
                signal = {
                    'model_dir': model_dir,
                    'model_chg_pct': round(model_pct, 2),
                    'mom60_dir': mom60_dir,
                    'ols60_dir': ols60_dir,
                    'agree_trend': bool(model_dir and mom60_dir and model_dir == mom60_dir),
                }
        except Exception:
            signal = None

        paper_record = None
        if mode == 'future':
            try:
                records = _load_paper_log()
                base_px = float(df['close'].iloc[-1])
                end_px = float(prediction_results[-1]['close'])
                pred_chg = (end_px - base_px) / base_px * 100.0
                rec = {
                    'id': datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'),
                    'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'file_path': file_path,
                    'symbol': os.path.splitext(os.path.basename(file_path))[0],
                    'model': CURRENT_MODEL_KEY or 'unknown',
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'last_real_date': str(df['timestamps'].iloc[-1])[:10],
                    'pred_end_date': str(future_timestamps[-1])[:10] if len(future_timestamps) else None,
                    'base_price': round(base_px, 4),
                    'pred_end_price': round(end_px, 4),
                    'pred_chg': round(pred_chg, 2),
                    'pred_dir': _direction(pred_chg),
                    'verified': False,
                }
                records.insert(0, rec)
                records = _verify_paper_records(records)
                _save_paper_log(records)
                paper_record = records[0] if records else rec
            except Exception as e:
                print(f"Failed to record paper-trading entry: {e}")

        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'summary': build_prediction_summary(prediction_results, x_df, pred_len, actual_data, is_forward=(mode == 'future')),
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'prediction_params': {
                'temperature': temperature,
                'top_p': top_p,
                'sample_count': sample_count,
                'mode': mode
            },
            'confidence': load_eval_cache(file_path, lookback, pred_len),
            'signal': signal,
            'paper_record': paper_record,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor, CURRENT_MODEL_KEY
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        CURRENT_MODEL_KEY = model_key
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070, use_reloader=False)
