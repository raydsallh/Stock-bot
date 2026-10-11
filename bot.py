import asyncio
import json
import logging
import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
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
    if df.empty or len(df) < min_rows:
        return None
    return df


def analyze(ticker: str, df: pd.DataFrame) -> dict:
    c = df["Close"]
    m, s = macd(c)
    r = rsi(c)
    return {
        "ticker": ticker,
        "price": c.iloc[-1],
        "change": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
        "rsi": r.iloc[-1],
        "macd_up": m.iloc[-1] > s.iloc[-1],
        "macd_cross": (m.iloc[-1] > s.iloc[-1]) != (m.iloc[-2] > s.iloc[-2]),
        "sma50": c.rolling(50).mean().iloc[-1],
        "sma200": c.rolling(200).mean().iloc[-1],
        "high52": df["High"].max(),
        "low52": df["Low"].min(),
    }


def fmt(a: dict) -> str:
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
    arrow = "📈" if a["change"] >= 0 else "📉"
    return (
        f"{arrow} {a['ticker']}: ${a['price']:.2f} ({a['change']:+.2f}%)\n"
        f"RSI: {a['rsi']:.1f}\n"
        f"أعلى/أدنى 52 أسبوع: ${a['high52']:.2f} / ${a['low52']:.2f}\n\n"
        + "\n".join(notes)
        + "\n\n⚠️ للمعلومات فقط، وليس توصية استثمارية."
    )


# ---------- الأوامر ----------
HELP = (
    "أرسل رمز سهم أمريكي مثل AAPL لتحليله.\n\n"
    "/price AAPL - سعر سريع\n"
    "/compare AAPL MSFT - مقارنة بين سهمين\n"
    "/top - أكثر الأسهم ارتفاعًا وانخفاضًا اليوم\n"
    "/alert AAPL 250 - نبّهني عند وصول السعر\n"
    "/alerts - عرض تنبيهاتي\n"
    "/unalert AAPL - إلغاء تنبيهات سهم\n\n"
    "/watch TSLA - أضف سهم لقائمتك\n"
    "/unwatch TSLA - احذفه\n"
    "/list - عرض القائمة\n"
    "/report - تحليل كل قائمتك الآن\n\n"
    "أرسل لك يوميًا بعد إغلاق السوق تنبيهًا إذا حصل شيء مهم في أسهمك."
)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)


async def reply_analysis(update: Update, ticker: str):
    ticker = ticker.strip().upper()
    df = await asyncio.to_thread(fetch, ticker)
    if df is None:
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {ticker}")
    await update.message.reply_text(fmt(analyze(ticker, df)))


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await reply_analysis(update, update.message.text)


# /price
async def price_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /price AAPL")
    t = ctx.args[0].upper()
    df = await asyncio.to_thread(fetch, t, "5d", 2)
    if df is None:
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {t}")
    c = df["Close"]
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    arrow = "📈" if chg >= 0 else "📉"
    await update.message.reply_text(f"{arrow} {t}: ${c.iloc[-1]:.2f} ({chg:+.2f}%)")


# /compare
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
    t1, t2 = ctx.args[0].upper(), ctx.args[1].upper()
    d1, d2 = await asyncio.gather(
        asyncio.to_thread(fetch, t1), asyncio.to_thread(fetch, t2)
    )
    if d1 is None or d2 is None:
        bad = t1 if d1 is None else t2
        return await update.message.reply_text(f"ما لقيت بيانات للرمز {bad}")
    a, b = snapshot(t1, d1), snapshot(t2, d2)
    text = (
        f"⚖️ {t1}  |  {t2}\n\n"
        + line("السعر", f"${a['price']:.2f}", f"${b['price']:.2f}") + "\n"
        + line("اليوم", f"{a['change']:+.2f}%", f"{b['change']:+.2f}%") + "\n"
        + line("عائد 3 أشهر", f"{a['ret3m']:+.1f}%", f"{b['ret3m']:+.1f}%") + "\n"
        + line("RSI", f"{a['rsi']:.0f}", f"{b['rsi']:.0f}") + "\n"
        + line("موقعه في نطاق 52 أسبوع", f"{a['pos52']:.0f}%", f"{b['pos52']:.0f}%") + "\n"
        + line("الاتجاه", trend(a), trend(b))
        + "\n\n⚠️ للمعلومات فقط، وليس توصية استثمارية."
    )
    await update.message.reply_text(text)


# /top
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


# /alert, /alerts, /unalert
async def alert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    usage = "مثال: /alert AAPL 250"
    if len(ctx.args) != 2:
        return await update.message.reply_text(usage)
    t = ctx.args[0].upper()
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
        f"تمام ✅ بنبّهك عند {word} ${target:.2f} (السعر الآن ${price:.2f})"
    )


async def alerts_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    items = alerts.get(str(update.effective_chat.id), [])
    if not items:
        return await update.message.reply_text("ما عندك تنبيهات. مثال: /alert AAPL 250")
    rows = [
        f"{i['t']} {'⬆️' if i['dir'] == 'above' else '⬇️'} ${i['target']:.2f}"
        for i in items
    ]
    await update.message.reply_text("تنبيهاتك:\n" + "\n".join(rows))


async def unalert_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /unalert AAPL")
    t = ctx.args[0].upper()
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
                        f"🔔 {item['t']} وصل ${price:.2f} (هدفك ${item['target']:.2f})",
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
        return await update.message.reply_text("مثال: /watch AAPL")
    t = ctx.args[0].upper()
    lst = watchlists.setdefault(str(update.effective_chat.id), [])
    if t not in lst:
        lst.append(t)
        save_watchlists()
    await update.message.reply_text(f"تمت إضافة {t} ✅")


async def unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("مثال: /unwatch AAPL")
    t = ctx.args[0].upper()
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
        await reply_analysis(update, t)


async def daily_alerts(ctx: ContextTypes.DEFAULT_TYPE):
    for chat_id, tickers in watchlists.items():
        for t in tickers:
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
    app.add_handler(MessageHandler(filters.Regex(r"^[A-Za-z.\-]{1,6}$"), on_text))
    # تنبيهات الأسعار كل 5 دقائق
    app.job_queue.run_repeating(check_alerts, interval=300, first=30)
    # التنبيه اليومي بعد إغلاق السوق الأمريكي، الاثنين إلى الجمعة
    # (في المكتبة: 0 = الأحد ... 6 = السبت)
    app.job_queue.run_daily(
        daily_alerts,
        time(16, 30, tzinfo=ZoneInfo("America/New_York")),
        days=(1, 2, 3, 4, 5),
    )
    app.run_polling()


if __name__ == "__main__":
    main()
