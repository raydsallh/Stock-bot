import asyncio
import json
import logging
import os
from datetime import time
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from telegram import Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

logging.basicConfig(level=logging.INFO)
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()

WATCH_FILE = Path("watchlists.json")
ALERT_FILE = Path("alerts.json")


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


watchlists: dict[str, list[str]] = load(WATCH_FILE)
alerts: dict[str, list[dict]] = load(ALERT_FILE)


def save_watchlists():
    WATCH_FILE.write_text(json.dumps(watchlists))


def save_alerts():
    ALERT_FILE.write_text(json.dumps(alerts))


# أكبر الشركات الأمريكية، تُستخدم في أمر /top
UNIVERSE = (
    "AAPL MSFT NVDA GOOGL AMZN META TSLA BRK-B AVGO JPM LLY V UNH XOM MA COST "
    "HD PG JNJ NFLX ORCL BAC ABBV CRM KO WMT AMD CVX MRK PEP ADBE TMO CSCO MCD "
    "ACN LIN DIS ABT QCOM INTC IBM PFE NKE"
).split()


def normalize(raw: str) -> str:
    """رقم سعودي مثل 2222 يتحول إلى 2222.SR، والباقي يبقى كما هو."""
    t = raw.strip().upper()
    return t + ".SR" if t.isdigit() else t


def money(v: float, ticker: str) -> str:
    return f"{v:.2f} ر.س" if ticker.endswith(".SR") else f"${v:.2f}"


# ---------- المؤشرات الفنية ----------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close: pd.Series):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def fetch(ticker: str, period: str = "1y", min_rows: int = 30):
    df = yf.Ticker(ticker).history(period=period)
    df = df.dropna(subset=["Close"])
    if df.empty or len(df) < min_rows:
        return None
    return df


MA_PERIODS = (5, 10, 20, 50, 100, 200)
STANCE_EN = ["Strongly negative", "Negative", "Neutral", "Positive", "Strongly positive"]
STANCE_AR = ["سلبية قوية", "سلبية", "محايدة", "إيجابية", "إيجابية قوية"]


def ma_stance(close: pd.Series):
    """كم متوسط متحرك السعر فوقه، ومنه الصورة الفنية العامة (0 إلى 4)."""
    above = avail = 0
    for p in MA_PERIODS:
        if len(close) >= p:
            avail += 1
            above += close.iloc[-1] > close.rolling(p).mean().iloc[-1]
    ratio = above / avail if avail else 0.5
    if ratio >= 0.85:
        idx = 4
    elif ratio >= 0.6:
        idx = 3
    elif ratio >= 0.4:
        idx = 2
    elif ratio >= 0.15:
        idx = 1
    else:
        idx = 0
    return int(above), avail, idx


def analyze(ticker: str, df: pd.DataFrame) -> dict:
    c = df["Close"]
    m, s = macd(c)
    r = rsi(c)
    year = df.tail(252)
    above, avail, idx = ma_stance(c)
    return {
        "ticker": ticker,
        "price": c.iloc[-1],
        "change": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
        "rsi": r.iloc[-1],
        "macd_up": m.iloc[-1] > s.iloc[-1],
        "macd_cross": (m.iloc[-1] > s.iloc[-1]) != (m.iloc[-2] > s.iloc[-2]),
        "sma50": c.rolling(50).mean().iloc[-1],
        "sma200": c.rolling(200).mean().iloc[-1],
        "high52": year["High"].max(),
        "low52": year["Low"].min(),
        "above": above,
        "avail": avail,
        "stance": idx,
    }


def fmt(a: dict) -> str:
    t = a["ticker"]
    notes = []
    if a["rsi"] >= 70:
        notes.append("🔴 RSI في منطقة تشبّع شراء")
    elif a["rsi"] <= 30:
        notes.append("🟢 RSI في منطقة تشبّع بيع")
    else:
        notes.append("⚪ RSI محايد")
    notes.append("🟢 MACD فوق خط الإشارة" if a["macd_up"] else "🔴 MACD تحت خط الإشارة")
    if a["macd_cross"]:
        notes.append("⚡ حدث تقاطع MACD اليوم")
    if pd.notna(a["sma50"]):
        notes.append("🟢 السعر فوق SM
