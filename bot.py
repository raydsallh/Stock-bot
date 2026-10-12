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
        notes.append("🟢 السعر فوق SMA50" if a["price"] > a["sma50"] else "🔴 السعر تحت SMA50")
    if pd.notna(a["sma200"]):
        notes.append("🟢 السعر فوق SMA200" if a["price"] > a["sma200"] else "🔴 السعر تحت SMA200")
    notes.append(
        f"📊 الصورة الفنية: {STANCE_AR[a['stance']]} "
        f"(السعر فوق {a['above']} من {a['avail']} متوسطات)"
    )
    arrow = "📈" if a["change"] >= 0 else "📉"
    return (
        f"{arrow} {t}: {money(a['price'], t)} ({a['change']:+.2f}%)\n"
        f"RSI: {a['rsi']:.1f}\n"
        f"أعلى/أدنى 52 أسبوع: {money(a['high52'], t)} / {money(a['low52'], t)}\n\n"
        + "\n".join(notes)
        + "\n\n⚠️ للمعلومات فقط، وليس توصية استثمارية."
    )


# ---------- مستويات الدعم والمقاومة ----------
def merge_levels(levels, tol: float = 0.012):
    out = []
    for v in sorted(levels):
        if out and abs(v - out[-1]) / out[-1] < tol:
            out[-1] = (out[-1] + v) / 2
        else:
            out.append(v)
    return out


def sr_levels(df: pd.DataFrame, window: int = 5, lookback: int = 250):
    """قمم وقيعان محلية قريبة من السعر الحالي (مستويات فنية وليست أهدافًا مضمونة)."""
    d = df.tail(lookback)
    hi, lo = d["High"].values, d["Low"].values
    price = d["Close"].iloc[-1]
    res, sup = [], []
    for i in range(window, len(d) - window):
        if hi[i] == hi[i - window:i + window + 1].max():
            res.append(hi[i])
        if lo[i] == lo[i - window:i + window + 1].min():
            sup.append(lo[i])
    res = [v for v in merge_levels(res) if v > price][:3]
    sup = [v for v in merge_levels(sup) if v < price][-2:][::-1]
    return res, sup


# ---------- ملف PDF: الشارت اليومي والأسبوعي والملخص ----------
UP, DOWN = "#1a9a3a", "#e02020"


def draw_candles(ax, v: pd.DataFrame):
    x = np.arange(len(v))
    o, h, l, c = (v[k].values for k in ("Open", "High", "Low", "Close"))
    col = [UP if cc >= oo else DOWN for oo, cc in zip(o, c)]
    ax.vlines(x, l, h, colors=col, linewidth=0.7)
    body = np.maximum(np.abs(c - o), (h.max() - l.min()) * 0.0015)
    ax.bar(x, body, bottom=np.minimum(o, c), color=col, width=0.65)


def chart_page(pdf, ticker: str, d: pd.DataFrame, label: str, bars: int, window: int):
    cur = "SAR" if ticker.endswith(".SR") else "USD"
    close = d["Close"]
    mas = {n: close.rolling(n).mean() for n in (7, 20, 50, 200)}
    m, s = macd(close)
    v = d.tail(bars)
    n = len(v)
    x = np.arange(n)
    res, sup = sr_levels(d, window=window, lookback=bars)

    fig = Figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[6, 1.4, 1.6], hspace=0.12,
        left=0.07, right=0.93, top=0.93, bottom=0.07,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    draw_candles(ax1, v)
    styles = {
        7: ("#000000", ":", 1.0),
        20: ("#f4a261", "--", 0.9),
        50: ("#1f4fd8", "-", 1.1),
        200: ("#ff8c00", "-", 1.1),
    }
    for k, (colr, ls, lw) in styles.items():
        ax1.plot(x, mas[k].tail(bars).values, color=colr, linestyle=ls, linewidth=lw, label=f"MA {k}")
    for i, lv in enumerate(res, 1):
        ax1.axhline(lv, color="#a23b9c", linestyle=":", linewidth=0.8)
        ax1.text(n + 0.5, lv, f"R{i}: {lv:.2f}", color="#a23b9c", fontsize=7, va="center")
    for i, lv in enumerate(sup, 1):
        ax1.axhline(lv, color="#2a6f97", linestyle=":", linewidth=0.8)
        ax1.text(n + 0.5, lv, f"S{i}: {lv:.2f}", color="#2a6f97", fontsize=7, va="center")
    ax1.set_xlim(-1, n + max(8, n * 0.12))
    lo_p, hi_p = v["Low"].min(), v["High"].max()
    pad = (hi_p - lo_p) * 0.05
    ax1.set_ylim(lo_p - pad, hi_p + pad)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9)
    last = v.iloc[-1]
    ax1.text(
        0.99, 0.985,
        f"O {last['Open']:.2f}   H {last['High']:.2f}   L {last['Low']:.2f}   C {last['Close']:.2f}",
        transform=ax1.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    fig.suptitle(
        f"{ticker}  -  {label}  ({v.index[-1].strftime('%Y-%m-%d')}, {cur})",
        fontsize=14, fontweight="bold", y=0.965,
    )

    up = (v["Close"] >= v["Open"]).values
    ax2.bar(x, v["Volume"].values, color=[UP if u else DOWN for u in up], width=0.65)
    ax2.set_ylabel("Volume", fontsize=8)
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda t, _: f"{t / 1e6:.1f}M" if t >= 1e6 else f"{t / 1e3:.0f}k")
    )

    mv, sv = m.tail(bars).values, s.tail(bars).values
    hist = mv - sv
    ax3.bar(x, hist, color=[UP if h >= 0 else DOWN for h in hist], width=0.65)
    ax3.plot(x, mv, color="#1f4fd8", linewidth=0.9)
    ax3.plot(x, sv, color=DOWN, linewidth=0.9)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("MACD", fontsize=8)

    ticks = np.linspace(0, n - 1, 8).astype(int)
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([v.index[i].strftime("%b %Y") for i in ticks], fontsize=7)
    for ax in (ax1, ax2):
        ax.tick_params(labelbottom=False)
    for ax in (ax1, ax2, ax3):
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_facecolor("#eef1f8")
        ax.grid(alpha=0.35)
    pdf.savefig(fig)


def summary_page(pdf, ticker: str, df: pd.DataFrame):
    a = analyze(ticker, df)
    c = df["Close"]
    cur = "SAR" if ticker.endswith(".SR") else "USD"
    res, sup = sr_levels(df)

    lines = [
        (f"Technical Summary - {ticker}", 16, "bold"),
        (f"Date: {df.index[-1].strftime('%Y-%m-%d')}", 10, "normal"),
        ("", 10, "normal"),
        (f"Price: {a['price']:.2f} {cur}   ({a['change']:+.2f}% today)", 11, "normal"),
        (f"52-week range: {a['low52']:.2f} - {a['high52']:.2f}", 11, "normal"),
    ]
    rsi_txt = "Overbought" if a["rsi"] >= 70 else "Oversold" if a["rsi"] <= 30 else "Neutral"
    lines.append((f"RSI(14): {a['rsi']:.1f}  ({rsi_txt})", 11, "normal"))
    macd_txt = "above signal line" if a["macd_up"] else "below signal line"
    if a["macd_cross"]:
        macd_txt += "  (crossover today)"
    lines.append((f"MACD: {macd_txt}", 11, "normal"))
    lines.append(("", 10, "normal"))
    lines.append(("Price vs moving averages", 12, "bold"))
    for p in MA_PERIODS:
        if len(c) >= p:
            ma = c.rolling(p).mean().iloc[-1]
            pos = "above" if c.iloc[-1] > ma else "below"
            lines.append((f"  MA {p:<4}{ma:>10.2f}   price is {pos}", 10, "normal"))
    lines.append(("", 10, "normal"))
    lines.append(
        (f"Trend picture: {STANCE_EN[a['stance']]}  "
         f"(price above {a['above']} of {a['avail']} averages)", 12, "bold")
    )
    lines.append(("", 10, "normal"))
    lines.append(("Nearby technical levels (recent swing highs / lows)", 12, "bold"))
    for i, lv in enumerate(res, 1):
        lines.append((f"  Resistance {i}: {lv:.2f}", 10, "normal"))
    for i, lv in enumerate(sup, 1):
        lines.append((f"  Support {i}: {lv:.2f}", 10, "normal"))
    lines.append(("", 10, "normal"))
    for t in (
        "These readings come from lagging technical indicators: they follow the",
        "market and do not predict future price moves. Levels are derived from",
        "recent price swings and are not price targets. For information only,",
        "not investment advice.",
    ):
        lines.append((t, 8.5, "normal"))

    fig = Figure(figsize=(8.27, 11.69))
    y = 0.94
    for text, size, weight in lines:
        fig.text(0.08, y, text, fontsize=size, fontweight=weight, family="monospace", va="top")
        y -= 0.006 + size * 0.0021
    pdf.savefig(fig)


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()


def make_pdf(ticker: str, df: pd.DataFrame) -> BytesIO:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        chart_page(pdf, ticker, df, "Daily Chart", min(len(df), 250), 5)
        wk = weekly(df)
        chart_page(pdf, ticker, wk, "Weekly Chart", min(len(wk), 180), 3)
        summary_page(pdf, ticker, df)
    buf.seek(0)
    return buf


# ---------- الأوامر ----------
HELP = (
    "أرسل رمز سهم أمريكي مثل AAPL، أو رقم سهم سعودي مثل 2222، "
    "وأرسل لك ملف PDF فيه الشارت اليومي والأسبوعي وملخص فني.\n\n"
    "/price AAPL - سعر سريع\n"
    "/compare AAPL MSFT - مقارنة بين سهمين\n"
    "/top - أكثر الأسهم ارتفاعًا وانخفاضًا اليوم (أمريكي)\n"
    "/alert AAPL 250 - نبّهني عند وصول السعر\n"
    "/alerts - عرض تنبيهاتي\n"
    "/unalert AAPL - إلغاء تنبيهات سهم\n\n"
    "/watch TSLA - أضف سهم لقائمتك\n"
    "/unwatch TSLA - احذفه\n"
    "/list - عرض القائمة\n"
    "/report - تحليل نصي لكل قائمتك\n\n"
    "أرسل لك يوميًا بعد إغلاق السوق تنبيهًا إذا حصل شيء مهم في أسهمك."
)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)


async def reply_analysis(update: Update, raw: str):
    ticker = normalize(raw)
    df = await asyncio.to_thread(fetch, ticker, "5y", 60)
    if df is None:
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {ticker}")
    a = analyze(ticker, df)
    pdf_buf = await asyncio.to_thread(make_pdf, ticker, df)
    await update.message.reply_document(
        document=pdf_buf, filename=f"{ticker}.pdf", caption=fmt(a)
    )


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await reply_analysis(update, update.message.text)


async def price_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /price AAPL أو /price 2222")
    t = normalize(ctx.args[0])
    df = await asyncio.to_thread(fetch, t, "5d", 2)
    if df is None:
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {t}")
    c = df["Close"]
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    arrow = "📈" if chg >= 0 else "📉"
    await update.message.reply_text(f"{arrow} {t}: {money(c.iloc[-1], t)} ({chg:+.2f}%)")


def line(label: str, a, b) -> str:
    return f"{label}: {a}  |  {b}"


def snapshot(t: str, df: pd.DataFrame) -> dict:
    a = analyze(t, df)
    c = df["Close"]
    a["ret3m"] = (c.iloc[-1] / c.iloc[-63] - 1) * 100 if len(c) > 63 else float("nan")
    rng = a["high52"] - a["low52"]
    a["pos52"] = (a["price"] - a["low52"]) / rng * 100 if rng else float("nan")
    return a


def trend(a: dict) -> str:
    if pd.isna(a["sma200"]):
        return "—"
    return "فوق SMA200 ✅" if a["price"] > a["sma200"] else "تحت SMA200 ❌"


async def compare_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) != 2:
        return await update.message.reply_text("مثال: /compare AAPL MSFT")
    t1, t2 = normalize(ctx.args[0]), normalize(ctx.args[1])
    d1, d2 = await asyncio.gather(
        asyncio.to_thread(fetch, t1), asyncio.to_thread(fetch, t2)
    )
    if d1 is None or d2 is None:
        bad = t1 if d1 is None else t2
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {bad}")
    a, b = snapshot(t1, d1), snapshot(t2, d2)
    text = (
        f"⚖️ {t1}  |  {t2}\n\n"
        + line("السعر", money(a["price"], t1), money(b["price"], t2)) + "\n"
        + line("اليوم", f"{a['change']:+.2f}%", f"{b['change']:+.2f}%") + "\n"
        + line("عائد 3 أشهر", f"{a['ret3m']:+.1f}%", f"{b['ret3m']:+.1f}%") + "\n"
        + line("RSI", f"{a['rsi']:.0f}", f"{b['rsi']:.0f}") + "\n"
        + line("موقعه في نطاق 52 أسبوع", f"{a['pos52']:.0f}%", f"{b['pos52']:.0f}%") + "\n"
        + line("الاتجاه", trend(a), trend(b))
        + "\n\n⚠️ للمعلومات فقط، وليس توصية استثمارية."
    )
    await update.message.reply_text(text)


def top_movers():
    data = yf.download(UNIVERSE, period="5d", progress=False)["Close"].ffill()
    chg = ((data.iloc[-1] / data.iloc[-2] - 1) * 100).dropna().sort_values()
    return chg.head(5), chg.tail(5)[::-1]


async def top_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("جاري الحساب...")
    try:
        losers, gainers = await asyncio.to_thread(top_movers)
    except Exception:
        logging.exception("top failed")
        return await update.message.reply_text("تعذّر جلب البيانات الآن، جرّب بعد قليل.")
    up = "\n".join(f"📈 {t}: {v:+.2f}%" for t, v in gainers.items())
    down = "\n".join(f"📉 {t}: {v:+.2f}%" for t, v in losers.items())
    await update.message.reply_text(
        f"الأكثر ارتفاعًا (من أكبر 44 شركة أمريكية):\n{up}\n\nالأكثر انخفاضًا:\n{down}"
    )


async def alert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    usage = "مثال: /alert AAPL 250 أو /alert 2222 28"
    if len(ctx.args) != 2:
        return await update.message.reply_text(usage)
    t = normalize(ctx.args[0])
    try:
        target = float(ctx.args[1])
    except ValueError:
        return await update.message.reply_text(usage)
    df = await asyncio.to_thread(fetch, t, "5d", 1)
    if df is None:
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {t}")
    price = df["Close"].iloc[-1]
    if abs(target - price) < 0.005:
        return await update.message.reply_text("السعر الحالي نفس الهدف تقريبًا.")
    direction = "above" if target > price else "below"
    alerts.setdefault(str(update.effective_chat.id), []).append(
        {"t": t, "target": target, "dir": direction}
    )
    save_alerts()
    word = "ارتفاعه إلى" if direction == "above" else "نزوله إلى"
    await update.message.reply_text(
        f"تمام ✅ بنبّهك عند {word} {money(target, t)} (السعر الآن {money(price, t)})"
    )


async def alerts_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    items = alerts.get(str(update.effective_chat.id), [])
    if not items:
        return await update.message.reply_text("ما عندك تنبيهات. مثال: /alert AAPL 250")
    rows = [
        f"{i['t']} {'⬆️' if i['dir'] == 'above' else '⬇️'} {money(i['target'], i['t'])}"
        for i in items
    ]
    await update.message.reply_text("تنبيهاتك:\n" + "\n".join(rows))


async def unalert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /unalert AAPL")
    t = normalize(ctx.args[0])
    key = str(update.effective_chat.id)
    before = len(alerts.get(key, []))
    alerts[key] = [i for i in alerts.get(key, []) if i["t"] != t]
    save_alerts()
    await update.message.reply_text(f"تم حذف {before - len(alerts[key])} تنبيه لـ {t}")


async def check_alerts(ctx: ContextTypes.DEFAULT_TYPE):
    changed = False
    for chat_id, items in list(alerts.items()):
        for item in list(items):
            try:
                df = await asyncio.to_thread(fetch, item["t"], "5d", 1)
                if df is None:
                    continue
                price = df["Close"].iloc[-1]
                hit = price >= item["target"] if item["dir"] == "above" else price <= item["target"]
                if hit:
                    await ctx.bot.send_message(
                        chat_id,
                        f"🔔 {item['t']} وصل {money(price, item['t'])} "
                        f"(هدفك {money(item['target'], item['t'])})",
                    )
                    items.remove(item)
                    changed = True
            except Exception:
                logging.exception("alert check failed")
    if changed:
        save_alerts()


# قائمة المتابعة
async def watch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /watch AAPL أو /watch 2222")
    t = normalize(ctx.args[0])
    lst = watchlists.setdefault(str(update.effective_chat.id), [])
    if t not in lst:
        lst.append(t)
        save_watchlists()
    await update.message.reply_text(f"تمت إضافة {t} ✅")


async def unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /unwatch AAPL")
    t = normalize(ctx.args[0])
    lst = watchlists.get(str(update.effective_chat.id), [])
    if t in lst:
        lst.remove(t)
        save_watchlists()
    await update.message.reply_text(f"تم حذف {t}")


async def list_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lst = watchlists.get(str(update.effective_chat.id), [])
    await update.message.reply_text("قائمتك: " + ", ".join(lst) if lst else "القائمة فاضية")


async def report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lst = watchlists.get(str(update.effective_chat.id), [])
    if not lst:
        return await update.message.reply_text("القائمة فاضية. استخدم /watch")
    for t in lst:
        df = await asyncio.to_thread(fetch, t)
        if df is None:
            await update.message.reply_text(f"ما لقيت بيانات للرمز {t}")
        else:
            await update.message.reply_text(fmt(analyze(t, df)))


async def daily_alerts(ctx: ContextTypes.DEFAULT_TYPE):
    saudi = ctx.job.data == "SA"
    for chat_id, tickers in watchlists.items():
        for t in tickers:
            if t.endswith(".SR") != saudi:
                continue
            df = await asyncio.to_thread(fetch, t)
            if df is None:
                continue
            a = analyze(t, df)
            if a["rsi"] >= 70 or a["rsi"] <= 30 or a["macd_cross"]:
                await ctx.bot.send_message(chat_id, "🔔 تنبيه\n" + fmt(a))


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("price", price_cmd))
    app.add_handler(CommandHandler("compare", compare_cmd))
    app.add_handler(CommandHandler("top", top_cmd))
    app.add_handler(CommandHandler("alert", alert_cmd))
    app.add_handler(CommandHandler("alerts", alerts_cmd))
    app.add_handler(CommandHandler("unalert", unalert_cmd))
    app.add_handler(CommandHandler("watch", watch))
    app.add_handler(CommandHandler("unwatch", unwatch))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(MessageHandler(filters.Regex(r"^[A-Za-z0-9.\-]{1,10}$"), on_text))
    # تنبيهات الأسعار كل 5 دقائق
    app.job_queue.run_repeating(check_alerts, interval=300, first=30)
    # أيام الأسبوع في المكتبة: 0 = الأحد ... 6 = السبت
    # السوق الأمريكي بعد الإغلاق (4:30 مساءً نيويورك)، الاثنين إلى الجمعة
    app.job_queue.run_daily(
        daily_alerts,
        time(16, 30, tzinfo=ZoneInfo("America/New_York")),
        days=(1, 2, 3, 4, 5),
        data="US",
    )
    # السوق السعودي بعد الإغلاق (3:30 مساءً الرياض)، الأحد إلى الخميس
    app.job_queue.run_daily(
        daily_alerts,
        time(15, 30, tzinfo=ZoneInfo("Asia/Riyadh")),
        days=(0, 1, 2, 3, 4),
        data="SA",
    )
    app.run_polling()


if __name__ == "__main__":
    main()
