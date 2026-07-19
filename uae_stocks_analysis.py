#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكربت تحليل أسهم الإمارات - نسخة موسّعة
=================================================================================
يغطي: Nadaraya-Watson, Linear Regression Channel, Fibonacci Bollinger Bands,
RSI, MACD, EMA20/50/200, VWAP, ATR, ADX, MFI, حجم التداول، الدعم والمقاومة،
الشموع اليابانية، هيكل الاتجاه (HH/HL/LH/LL)، إجماع المحللين، الأرباح
والتوزيعات، الأخبار الجوهرية، ونسبة المخاطرة إلى العائد.

المتطلبات:
    pip install yfinance pandas numpy matplotlib

الاستخدام:
    python uae_stocks_analysis.py
    python uae_stocks_analysis.py EMAAR.AE ETISALAT.AB --plot --csv
    python uae_stocks_analysis.py --days 400
    python uae_stocks_analysis.py --backtest                      # محاكاة تاريخية للاستراتيجية
    python uae_stocks_analysis.py EMAAR.AE --backtest --entry-threshold 0.2

رموز الأسهم على Yahoo Finance:
    - سوق دبي المالي (DFM) -> تنتهي بـ .AE   مثل EMAAR.AE
    - سوق أبوظبي (ADX)     -> تنتهي بـ .AB   مثل ADCB.AB
"""

import sys
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("مكتبة yfinance غير مثبتة. شغّل:\n    pip install yfinance pandas numpy matplotlib")
    sys.exit(1)


DEFAULT_TICKERS = [
    "ADNOCGAS.AB",     # 1. أدنوك للغاز - أبوظبي
    "ETISALAT.AB",     # 2. اتصالات e& - أبوظبي
    "ADNOCDIST.AB",    # 3. أدنوك للتوزيع - أبوظبي
    "ADNOCDRILL.AB",   # 4. أدنوك للحفر - أبوظبي
    "SALIK.AE",        # 5. سالك - دبي
    "EMAAR.AE",        # 6. إعمار العقارية - دبي
    "DEWA.AE",         # 7. هيئة كهرباء ومياه دبي - دبي
]

# رموز المؤشرات العامة للسوقين (لقياس هل حركة السهم بسبب السهم نفسه أو السوق كله)
MARKET_INDICES = {
    "مؤشر دبي العام (DFM General)": "DFMGI.AE",
    "مؤشر أبوظبي العام (ADX General)": "FADGI.FGI",
}


# وزن كل مؤشر (نجوم المستخدم) يُستخدم في حساب الإشارة المركّبة النهائية
WEIGHTS = {
    "nadaraya_watson": 5,
    "lin_reg_channel": 5,
    "fib_bb": 5,
    "rsi": 4,
    "macd": 4,
    "ema20": 3,
    "ema50": 4,
    "ema200": 5,
    "vwap": 4,
    "adx_di": 4,
    "mfi": 4,
    "volume": 5,
    "support_resistance": 5,
    "candlestick": 4,
    "trend_structure": 5,
}


# ============================== المؤشرات الأساسية ============================== #

def sma(series, window):
    return series.rolling(window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rma(series, period):
    """تنعيم وايلدر - يُستخدم في ATR/ADX."""
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(series, window=20, num_std=2):
    mid = sma(series, window)
    std = series.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return rma(tr, period)


def adx_indicator(df, period=14):
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr_smoothed = atr(df, period)
    plus_di = 100 * rma(plus_dm, period) / tr_smoothed
    minus_di = 100 * rma(minus_dm, period) / tr_smoothed
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return rma(dx, period), plus_di, minus_di


def mfi_indicator(df, period=14):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical * df["Volume"]
    delta = typical.diff()
    pos_flow = money_flow.where(delta > 0, 0.0).rolling(period).sum()
    neg_flow = money_flow.where(delta < 0, 0.0).rolling(period).sum()
    mfr = pos_flow / neg_flow
    return 100 - (100 / (1 + mfr))


def vwap_cumulative(df):
    """VWAP تراكمي منذ بداية الفترة المحمّلة (تقريب مناسب للبيانات اليومية)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()


def linear_regression_channel(series, window=50, num_std=2):
    y = series.iloc[-window:].values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    resid_std = (y - fit).std()
    return slope, fit[-1], fit[-1] + num_std * resid_std, fit[-1] - num_std * resid_std


def nadaraya_watson_envelope(series, window=50, bandwidth=8, mult=3):
    """نسخة غير مُعاد رسمها (non-repainting): كل نقطة تُحسب فقط من البيانات
    السابقة لها (نافذة سببية causal)، بدون استخدام بيانات مستقبلية."""
    values = series.values
    n = len(values)
    est = np.full(n, np.nan)
    kernel_idx = np.arange(window)
    weights = np.exp(-((window - 1 - kernel_idx) ** 2) / (2 * bandwidth ** 2))
    for i in range(window, n):
        est[i] = np.sum(weights * values[i - window:i]) / weights.sum()
    est_series = pd.Series(est, index=series.index)
    mae = (series - est_series).abs().rolling(window).mean()
    return est_series, est_series + mae * mult, est_series - mae * mult


def fibonacci_bollinger_bands(series, window=20):
    """تقريب شائع لـ Fibonacci Bollinger Bands: خط أساس SMA، ونطاقات
    بمضاعفات فيبوناتشي لانحراف معياري مضاعف بدل مضاعف بولينجر الثابت (2)."""
    basis = sma(series, window)
    dev = series.rolling(window).std() * 3
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
    upper = {f: basis + dev * f for f in fib_ratios}
    lower = {f: basis - dev * f for f in fib_ratios}
    return basis, upper, lower


def rolling_linreg(series, window=50, num_std=2):
    """نسخة (rolling) من قناة الانحدار الخطي تُرجع مصفوفة لكل يوم، بحيث كل
    نقطة تعتمد فقط على النافذة السابقة لها مباشرة (سببية بالكامل - تصلح للمحاكاة التاريخية)."""
    values = series.values
    n = len(values)
    slope_arr = np.full(n, np.nan)
    fit_arr = np.full(n, np.nan)
    upper_arr = np.full(n, np.nan)
    lower_arr = np.full(n, np.nan)
    x = np.arange(window)
    for i in range(window, n):
        y = values[i - window:i]
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * x + intercept
        resid_std = (y - fit).std()
        slope_arr[i] = slope
        fit_arr[i] = fit[-1]
        upper_arr[i] = fit[-1] + num_std * resid_std
        lower_arr[i] = fit[-1] - num_std * resid_std
    return slope_arr, fit_arr, upper_arr, lower_arr


def find_swing_points_positions(df, order=5):
    """مثل find_swing_points لكن تُرجع رقم الموضع (index position) بدل التاريخ،
    لتسهيل التحقق من "متى صارت القمة/القاع مؤكدة" أثناء المحاكاة التاريخية."""
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(order, n - order):
        if highs[i] == highs[i - order:i + order + 1].max():
            swing_highs.append((i, highs[i]))
        if lows[i] == lows[i - order:i + order + 1].min():
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def candlestick_signal(o, h, l, c, po, pc):
    """نسخة رقمية سريعة من detect_candlestick لاستخدامها داخل حلقة المحاكاة التاريخية."""
    body = abs(c - o)
    rng = h - l if h != l else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body / rng < 0.1:
        return 0
    if c > o and c > po and o < pc and body > abs(pc - po):
        return 1
    if c < o and c < po and o > pc and body > abs(pc - po):
        return -1
    if lower_wick > 2 * body and upper_wick < body:
        return 1
    if upper_wick > 2 * body and lower_wick < body:
        return -1
    return 0


# ============================== أدوات هيكلية ============================== #

def find_swing_points(df, order=5):
    highs, lows = df["High"], df["Low"]
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(order, n - order):
        wh = highs.iloc[i - order:i + order + 1]
        wl = lows.iloc[i - order:i + order + 1]
        if highs.iloc[i] == wh.max():
            swing_highs.append((df.index[i], highs.iloc[i]))
        if lows.iloc[i] == wl.min():
            swing_lows.append((df.index[i], lows.iloc[i]))
    return swing_highs, swing_lows


def support_resistance_levels(df, order=5, n_levels=3):
    swing_highs, swing_lows = find_swing_points(df, order)
    price = df["Close"].iloc[-1]
    resistances = sorted([lvl for _, lvl in swing_highs if lvl > price])[:n_levels]
    supports = sorted([lvl for _, lvl in swing_lows if lvl < price], reverse=True)[:n_levels]
    return supports, resistances


def trend_structure(df, order=5, n_swings=4):
    swing_highs, swing_lows = find_swing_points(df, order)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "غير كافٍ لتحديد الهيكل", 0
    last_highs = [h for _, h in swing_highs[-n_swings:]]
    last_lows = [l for _, l in swing_lows[-n_swings:]]
    higher_highs = all(x < y for x, y in zip(last_highs, last_highs[1:]))
    higher_lows = all(x < y for x, y in zip(last_lows, last_lows[1:]))
    lower_highs = all(x > y for x, y in zip(last_highs, last_highs[1:]))
    lower_lows = all(x > y for x, y in zip(last_lows, last_lows[1:]))

    if higher_highs and higher_lows:
        return "اتجاه صاعد واضح (Higher Highs / Higher Lows)", 1
    if lower_highs and lower_lows:
        return "اتجاه هابط واضح (Lower Highs / Lower Lows)", -1
    return "هيكل متذبذب / غير محدد الاتجاه", 0


def detect_candlestick(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = last["Open"], last["High"], last["Low"], last["Close"]
    po, pc = prev["Open"], prev["Close"]
    body = abs(c - o)
    rng = h - l if h != l else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if body / rng < 0.1:
        return "دوجي (Doji) - تردد في السوق", 0
    if c > o and c > po and o < pc and body > abs(pc - po):
        return "ابتلاع صاعد (Bullish Engulfing)", 1
    if c < o and c < po and o > pc and body > abs(pc - po):
        return "ابتلاع هابط (Bearish Engulfing)", -1
    if lower_wick > 2 * body and upper_wick < body:
        return "مطرقة (Hammer) - احتمال ارتداد صاعد", 1
    if upper_wick > 2 * body and lower_wick < body:
        return "نجمة الرماية (Shooting Star) - احتمال ارتداد هابط", -1
    return "شمعة عادية بدون نمط واضح", 0


# ============================== بيانات إضافية (اختيارية) ============================== #

def fetch_fundamental_info(ticker_obj):
    """يحاول جلب إجماع المحللين، الأرباح، التوزيعات، والأخبار. بعض هذه
    البيانات قد لا تتوفر لأسهم سوق الإمارات على Yahoo Finance."""
    result = {}
    try:
        info = ticker_obj.get_info()
    except Exception:
        info = {}
    result["recommendation"] = info.get("recommendationKey")
    result["target_mean"] = info.get("targetMeanPrice")
    result["target_high"] = info.get("targetHighPrice")
    result["target_low"] = info.get("targetLowPrice")
    result["num_analysts"] = info.get("numberOfAnalystOpinions")
    result["dividend_yield"] = info.get("dividendYield")

    try:
        divs = ticker_obj.dividends
        result["last_dividends"] = divs.tail(3) if not divs.empty else None
    except Exception:
        result["last_dividends"] = None

    try:
        earnings_dates = ticker_obj.get_earnings_dates(limit=4)
        result["earnings_dates"] = earnings_dates
    except Exception:
        result["earnings_dates"] = None

    try:
        news = ticker_obj.news
        result["news"] = news[:3] if news else []
    except Exception:
        result["news"] = []

    return result


# ============================== التحليل الرئيسي ============================== #

def fetch_latest_quote(ticker_obj):
    """يحاول جلب أحدث نقطة سعر متاحة فعلياً (بيانات دقيقة إن وُجدت لهذا السهم
    على Yahoo، وإلا آخر إغلاق يومي)، مع طابعها الزمني الحقيقي حتى يتضح للمستخدم
    بوضوح هل هذا فعلاً "سعر اليوم" أم آخر جلسة تداول سابقة."""
    try:
        intraday = ticker_obj.history(period="1d", interval="1m")
    except Exception:
        intraday = None

    if intraday is not None and not intraday.empty:
        return {
            "price": intraday["Close"].iloc[-1],
            "timestamp": intraday.index[-1],
            "source": "بيانات دقيقة (قريبة من اللحظي، عادة بتأخير 15-20 دقيقة)",
        }

    try:
        daily = ticker_obj.history(period="5d")
    except Exception:
        daily = None

    if daily is not None and not daily.empty:
        return {
            "price": daily["Close"].iloc[-1],
            "timestamp": daily.index[-1],
            "source": "آخر إغلاق يومي متاح (لا توجد بيانات داخل اليوم لهذا السهم على Yahoo)",
        }

    return None


def analyze_ticker(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        print(f"⚠️  لا توجد بيانات للرمز: {ticker}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["EMA200"] = ema(close, 200)
    df["SMA20"] = sma(close, 20)
    df["RSI14"] = rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = bollinger_bands(close)
    df["ATR14"] = atr(df)
    df["ADX"], df["Plus_DI"], df["Minus_DI"] = adx_indicator(df)
    df["MFI14"] = mfi_indicator(df)
    df["VWAP"] = vwap_cumulative(df)
    df["Vol_Avg20"] = df["Volume"].rolling(20).mean()

    return df


def build_composite_signal(df, ticker):
    last = df.iloc[-1]
    price = last["Close"]
    signals = {}
    notes = []

    signals["ema20"] = 1 if price > last["EMA20"] else -1
    signals["ema50"] = 1 if price > last["EMA50"] else -1
    signals["ema200"] = 1 if price > last["EMA200"] else -1
    notes.append(f"EMA20/50/200: {last['EMA20']:.2f} / {last['EMA50']:.2f} / {last['EMA200']:.2f}")

    if last["RSI14"] < 30:
        signals["rsi"] = 1
    elif last["RSI14"] > 70:
        signals["rsi"] = -1
    else:
        signals["rsi"] = 0
    notes.append(f"RSI(14): {last['RSI14']:.1f}")

    signals["macd"] = 1 if last["MACD"] > last["MACD_Signal"] else -1
    notes.append(f"MACD: {last['MACD']:.4f} مقابل الإشارة {last['MACD_Signal']:.4f}")

    if last["ADX"] > 20:
        signals["adx_di"] = 1 if last["Plus_DI"] > last["Minus_DI"] else -1
    else:
        signals["adx_di"] = 0
    notes.append(f"ADX: {last['ADX']:.1f} (قوة الاتجاه) | +DI {last['Plus_DI']:.1f} / -DI {last['Minus_DI']:.1f}")

    if last["MFI14"] < 20:
        signals["mfi"] = 1
    elif last["MFI14"] > 80:
        signals["mfi"] = -1
    else:
        signals["mfi"] = 0
    notes.append(f"MFI(14): {last['MFI14']:.1f}")

    vol_ratio = last["Volume"] / last["Vol_Avg20"] if last["Vol_Avg20"] > 0 else 1
    price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
    if vol_ratio > 1.2:
        signals["volume"] = 1 if price_change > 0 else -1
    else:
        signals["volume"] = 0
    notes.append(f"الحجم مقارنة بمتوسط 20 يوم: {vol_ratio:.2f}x")

    signals["vwap"] = 1 if price > last["VWAP"] else -1
    notes.append(f"VWAP: {last['VWAP']:.2f}")

    supports, resistances = support_resistance_levels(df)
    if supports and abs(price - supports[0]) / price < 0.015:
        signals["support_resistance"] = 1
        sr_note = f"السعر قريب من دعم عند {supports[0]:.2f}"
    elif resistances and abs(price - resistances[0]) / price < 0.015:
        signals["support_resistance"] = -1
        sr_note = f"السعر قريب من مقاومة عند {resistances[0]:.2f}"
    else:
        signals["support_resistance"] = 0
        sr_note = "السعر بين الدعم والمقاومة الحاليين"
    notes.append(sr_note)
    notes.append(f"أقرب دعوم: {[round(s,2) for s in supports]} | أقرب مقاومات: {[round(r,2) for r in resistances]}")

    candle_desc, candle_signal = detect_candlestick(df)
    signals["candlestick"] = candle_signal
    notes.append(f"الشمعة الأخيرة: {candle_desc}")

    structure_desc, structure_signal = trend_structure(df)
    signals["trend_structure"] = structure_signal
    notes.append(f"هيكل الاتجاه: {structure_desc}")

    slope, lr_value, lr_upper, lr_lower = linear_regression_channel(df["Close"])
    signals["lin_reg_channel"] = 1 if slope > 0 else -1
    notes.append(f"قناة الانحدار الخطي: الميل {'صاعد' if slope>0 else 'هابط'}, القناة [{lr_lower:.2f} - {lr_upper:.2f}]")

    nw_est, nw_upper, nw_lower = nadaraya_watson_envelope(df["Close"])
    if not np.isnan(nw_upper.iloc[-1]):
        if price < nw_lower.iloc[-1]:
            signals["nadaraya_watson"] = 1
        elif price > nw_upper.iloc[-1]:
            signals["nadaraya_watson"] = -1
        else:
            signals["nadaraya_watson"] = 0
        notes.append(f"Nadaraya-Watson: القيمة المقدرة {nw_est.iloc[-1]:.2f}, النطاق [{nw_lower.iloc[-1]:.2f} - {nw_upper.iloc[-1]:.2f}]")
    else:
        signals["nadaraya_watson"] = 0
        notes.append("Nadaraya-Watson: بيانات غير كافية بعد")

    fib_basis, fib_upper, fib_lower = fibonacci_bollinger_bands(df["Close"])
    fu_1 = fib_upper[1.0].iloc[-1]
    fl_1 = fib_lower[1.0].iloc[-1]
    if not np.isnan(fu_1):
        if price <= fib_lower[0.382].iloc[-1]:
            signals["fib_bb"] = 1
        elif price >= fib_upper[0.382].iloc[-1]:
            signals["fib_bb"] = -1
        else:
            signals["fib_bb"] = 0
        notes.append(f"Fibonacci Bollinger Bands: أساس {fib_basis.iloc[-1]:.2f}, نطاق كامل [{fl_1:.2f} - {fu_1:.2f}]")
    else:
        signals["fib_bb"] = 0

    notes.append(f"ATR(14): {last['ATR14']:.3f} (مقياس التذبذب - يُستخدم لتحديد حجم وقف الخسارة)")

    total_weight = sum(WEIGHTS[k] for k in signals)
    weighted_sum = sum(WEIGHTS[k] * v for k, v in signals.items())
    score = weighted_sum / total_weight  # بين -1 و 1

    if score >= 0.35:
        verdict = "📈 إشارة إيجابية قوية"
    elif score >= 0.1:
        verdict = "📈 إشارة إيجابية"
    elif score <= -0.35:
        verdict = "📉 إشارة سلبية قوية"
    elif score <= -0.1:
        verdict = "📉 إشارة سلبية"
    else:
        verdict = "➖ محايدة"

    # نسبة المخاطرة إلى العائد بناءً على أقرب دعم/مقاومة
    rr_note = "غير متاحة (لا يوجد دعم أو مقاومة واضحة قريبة)"
    if supports and resistances:
        risk = price - supports[0]
        reward = resistances[0] - price
        if risk > 0:
            rr = reward / risk
            rr_note = f"1 : {rr:.2f}  (وقف مقترح {supports[0]:.2f} | هدف مقترح {resistances[0]:.2f})"

    extra = {
        "price": price,
        "supports": supports,
        "resistances": resistances,
        "nw_lower": nw_lower.iloc[-1] if not np.isnan(nw_upper.iloc[-1]) else None,
        "nw_upper": nw_upper.iloc[-1] if not np.isnan(nw_upper.iloc[-1]) else None,
    }
    return score, verdict, notes, rr_note, extra


def print_fundamentals(fund):
    print("\n--- إجماع المحللين والبيانات الأساسية ---")
    if fund["recommendation"]:
        print(f"توصية المحللين: {fund['recommendation']}")
    if fund["target_mean"]:
        print(f"متوسط السعر المستهدف: {fund['target_mean']:.2f} "
              f"(نطاق {fund['target_low']:.2f} - {fund['target_high']:.2f}, عدد المحللين: {fund['num_analysts']})")
    if not fund["recommendation"] and not fund["target_mean"]:
        print("لا تتوفر بيانات تحليل محللين لهذا السهم على Yahoo Finance حالياً.")

    if fund["dividend_yield"]:
        print(f"عائد التوزيعات: {fund['dividend_yield']*100:.2f}%")
    if fund["last_dividends"] is not None and len(fund["last_dividends"]) > 0:
        print("آخر توزيعات:")
        for date, val in fund["last_dividends"].items():
            print(f"  {date.date()}: {val:.3f}")

    if fund["earnings_dates"] is not None and len(fund["earnings_dates"]) > 0:
        print("مواعيد الأرباح القادمة/الأخيرة:")
        print(fund["earnings_dates"].head(3).to_string())

    if fund["news"]:
        print("آخر الأخبار (العناوين فقط - راجع الرابط للتفاصيل):")
        for item in fund["news"]:
            title = item.get("title") or item.get("content", {}).get("title")
            link = item.get("link") or item.get("content", {}).get("clickThroughUrl", {}).get("url")
            if title:
                print(f"  • {title}")
                if link:
                    print(f"    {link}")
    else:
        print("لا توجد أخبار حديثة متاحة عبر الـ API لهذا السهم.")


def decision_from_score(score):
    if score >= 0.35:
        return "شراء قوي"
    elif score >= 0.1:
        return "شراء تدريجي"
    elif score <= -0.35:
        return "تجنب / بيع"
    elif score <= -0.1:
        return "تقليل / حذر"
    return "انتظار ومراقبة"


def print_decision_summary(ticker, score, verdict, extra, fund, amount):
    price = extra["price"]
    supports = extra["supports"]
    resistances = extra["resistances"]

    # منطقة الشراء المقترحة: بين أقرب دعم (أو النطاق السفلي لـ Nadaraya-Watson إن كان أقرب) والسعر الحالي
    zone_low_candidates = [v for v in [supports[0] if supports else None, extra["nw_lower"]] if v is not None]
    zone_low = max(zone_low_candidates) if zone_low_candidates else price * 0.97
    buy_zone = f"{zone_low:.2f} - {price:.2f}"

    # السعر المستهدف (12 شهر): تقديرات المحللين إن توفرت، وإلا أقرب مقاومة بعيدة كتقدير فني تقريبي
    if fund.get("target_mean"):
        target = fund["target_mean"]
        target_source = f"(إجماع {fund.get('num_analysts') or '؟'} محلل)"
    elif resistances:
        target = resistances[-1]
        target_source = "(تقدير فني تقريبي من أبعد مقاومة، وليس تقديراً مالياً لـ 12 شهر)"
    else:
        target = price * 1.10
        target_source = "(تقدير عام تقريبي - لا تتوفر بيانات كافية)"
    upside_pct = (target - price) / price * 100

    shares = int(amount // price) if price > 0 else 0
    actual_cost = shares * price
    leftover = amount - actual_cost

    recommendation = fund.get("recommendation") or "غير متوفرة"

    print("\n╔══════════════ ملخص القرار ══════════════╗")
    print(f"  السعر الحالي:            {price:.2f} AED")
    print(f"  إجماع المحللين:          {recommendation}")
    print(f"  درجة الإشارة (Score):    {score:+.2f}  -  {verdict}")
    print(f"  القرار:                  {decision_from_score(score)}")
    print(f"  منطقة الشراء المقترحة:   {buy_zone} AED")
    print(f"  السعر المستهدف (12 شهر): {target:.2f} AED  {target_source}")
    print(f"  نسبة الصعود المحتملة:    {upside_pct:+.1f}%")
    print(f"  المبلغ المخصص:           {amount:,.0f} AED")
    print(f"  عدد الأسهم الممكن شراؤها: {shares:,}  (تكلفة فعلية {actual_cost:,.0f} AED، متبقي {leftover:,.0f} AED)")

    news = fund.get("news") or []
    print("  أهم الأخبار:")
    if news:
        for item in news[:2]:
            title = item.get("title") or item.get("content", {}).get("title")
            if title:
                print(f"    • {title}")
    else:
        print("    لا تتوفر أخبار حديثة عبر الـ API لهذا السهم.")
    print("╚═══════════════════════════════════════════╝")


def backtest_strategy(df, order=5, entry_threshold=0.15, exit_threshold=-0.05,
                       stop_atr_mult=2.0, target_atr_mult=4.0):
    """
    محاكاة تاريخية لاستراتيجية Long-only بسيطة مبنية على نفس الإشارة المركّبة
    المستخدمة في التحليل الحي، لكن بحساب كل مؤشر بشكل سببي بحت (كل يوم يعتمد
    فقط على بيانات الماضي حتى ذلك اليوم) لتفادي Look-ahead Bias.

    قواعد الصفقة:
      - دخول: أول يوم يتجاوز فيه Score قيمة entry_threshold وما فيه صفقة مفتوحة.
      - خروج: أول ما يحصل من: وقف خسارة (سعر الدخول - ATR×stop_atr_mult)،
        أو هدف ربح (سعر الدخول + ATR×target_atr_mult)، أو انعكاس الإشارة تحت exit_threshold.
      - بدون عمولة أو انزلاق سعري (slippage) - النتائج الفعلية غالباً أضعف قليلاً من المحاكاة.
    """
    n = len(df)
    close, open_ = df["Close"].values, df["Open"].values
    high, low = df["High"].values, df["Low"].values

    ema20, ema50, ema200 = df["EMA20"].values, df["EMA50"].values, df["EMA200"].values
    rsi14 = df["RSI14"].values
    macd_l, macd_s = df["MACD"].values, df["MACD_Signal"].values
    adx_v, plus_di, minus_di = df["ADX"].values, df["Plus_DI"].values, df["Minus_DI"].values
    mfi14 = df["MFI14"].values
    vwap = df["VWAP"].values
    vol, vol_avg20 = df["Volume"].values, df["Vol_Avg20"].values
    atr14 = df["ATR14"].values

    _, nw_upper_s, nw_lower_s = nadaraya_watson_envelope(df["Close"])
    nw_upper, nw_lower = nw_upper_s.values, nw_lower_s.values

    _, fib_upper_d, fib_lower_d = fibonacci_bollinger_bands(df["Close"])
    fib_u38, fib_l38 = fib_upper_d[0.382].values, fib_lower_d[0.382].values

    slope_arr, _, _, _ = rolling_linreg(df["Close"])
    swing_highs_pos, swing_lows_pos = find_swing_points_positions(df, order)

    trades = []
    position = None
    start_idx = 200  # حتى يكتمل EMA200

    for t in range(start_idx, n):
        price = close[t]

        confirmed_highs = [lvl for i, lvl in swing_highs_pos if i + order <= t and lvl > price]
        confirmed_lows = [lvl for i, lvl in swing_lows_pos if i + order <= t and lvl < price]
        resistances = sorted(confirmed_highs)[:3]
        supports = sorted(confirmed_lows, reverse=True)[:3]

        signals = {
            "ema20": 1 if price > ema20[t] else -1,
            "ema50": 1 if price > ema50[t] else -1,
            "ema200": 1 if price > ema200[t] else -1,
            "rsi": 1 if rsi14[t] < 30 else (-1 if rsi14[t] > 70 else 0),
            "macd": 1 if macd_l[t] > macd_s[t] else -1,
            "adx_di": (1 if plus_di[t] > minus_di[t] else -1) if adx_v[t] > 20 else 0,
            "mfi": 1 if mfi14[t] < 20 else (-1 if mfi14[t] > 80 else 0),
            "vwap": 1 if price > vwap[t] else -1,
        }

        vol_ratio = vol[t] / vol_avg20[t] if vol_avg20[t] > 0 else 1
        price_chg = close[t] - close[t - 1]
        signals["volume"] = (1 if price_chg > 0 else -1) if vol_ratio > 1.2 else 0

        if supports and abs(price - supports[0]) / price < 0.015:
            signals["support_resistance"] = 1
        elif resistances and abs(price - resistances[0]) / price < 0.015:
            signals["support_resistance"] = -1
        else:
            signals["support_resistance"] = 0

        signals["candlestick"] = candlestick_signal(open_[t], high[t], low[t], close[t], open_[t - 1], close[t - 1])

        conf_h = [(i, lvl) for i, lvl in swing_highs_pos if i + order <= t]
        conf_l = [(i, lvl) for i, lvl in swing_lows_pos if i + order <= t]
        if len(conf_h) >= 2 and len(conf_l) >= 2:
            hh, hl = conf_h[-1][1] > conf_h[-2][1], conf_l[-1][1] > conf_l[-2][1]
            lh, ll = conf_h[-1][1] < conf_h[-2][1], conf_l[-1][1] < conf_l[-2][1]
            signals["trend_structure"] = 1 if (hh and hl) else (-1 if (lh and ll) else 0)
        else:
            signals["trend_structure"] = 0

        signals["lin_reg_channel"] = (1 if slope_arr[t] > 0 else -1) if not np.isnan(slope_arr[t]) else 0
        signals["nadaraya_watson"] = (1 if price < nw_lower[t] else (-1 if price > nw_upper[t] else 0)) if not np.isnan(nw_upper[t]) else 0
        signals["fib_bb"] = (1 if price <= fib_l38[t] else (-1 if price >= fib_u38[t] else 0)) if not np.isnan(fib_u38[t]) else 0

        total_w = sum(WEIGHTS[k] for k in signals)
        score = sum(WEIGHTS[k] * v for k, v in signals.items()) / total_w

        if position is None:
            if score >= entry_threshold:
                position = {
                    "entry_idx": t, "entry_price": price,
                    "stop": price - atr14[t] * stop_atr_mult,
                    "target": price + atr14[t] * target_atr_mult,
                }
        else:
            exit_price, exit_reason = None, None
            if low[t] <= position["stop"]:
                exit_price, exit_reason = position["stop"], "وقف خسارة"
            elif high[t] >= position["target"]:
                exit_price, exit_reason = position["target"], "هدف ربح"
            elif score <= exit_threshold:
                exit_price, exit_reason = price, "الإشارة انعكست"

            if exit_reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                trades.append({
                    "entry_date": df.index[position["entry_idx"]], "entry_price": position["entry_price"],
                    "exit_date": df.index[t], "exit_price": exit_price,
                    "return_pct": ret_pct, "reason": exit_reason,
                    "days_held": t - position["entry_idx"],
                })
                position = None

    if position is not None:
        t = n - 1
        ret_pct = (close[t] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            "entry_date": df.index[position["entry_idx"]], "entry_price": position["entry_price"],
            "exit_date": df.index[t], "exit_price": close[t],
            "return_pct": ret_pct, "reason": "نهاية الفترة (صفقة مفتوحة)",
            "days_held": t - position["entry_idx"],
        })

    return trades


def summarize_backtest(trades, df, start_idx=200):
    if not trades:
        return None
    returns = [tr["return_pct"] for tr in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    compounded = 1.0
    for r in returns:
        compounded *= (1 + r / 100)

    bh_start = df["Close"].iloc[start_idx]
    bh_end = df["Close"].iloc[-1]
    bh_return = (bh_end - bh_start) / bh_start * 100

    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(returns) * 100,
        "avg_win_pct": np.mean(wins) if wins else 0,
        "avg_loss_pct": np.mean(losses) if losses else 0,
        "total_return_pct": (compounded - 1) * 100,
        "buy_hold_return_pct": bh_return,
        "avg_days_held": np.mean([tr["days_held"] for tr in trades]),
        "profit_factor": profit_factor,
    }


def short_term_signal_row(df):
    """يبني إشارات قصيرة المدى بنفس أسلوب تطبيقات الوسطاء (Buy/Sell/Hold) لكل مؤشر."""
    last = df.iloc[-1]

    if last["RSI14"] < 30:
        rsi_sig = "Buy"
    elif last["RSI14"] > 70:
        rsi_sig = "Sell"
    else:
        rsi_sig = "Hold"

    ma20_sig = "Buy" if last["Close"] > last["EMA20"] else "Sell"
    ma_cross_sig = "Buy" if last["EMA20"] > last["EMA50"] else "Sell"
    macd_sig = "Buy" if last["MACD"] > last["MACD_Signal"] else "Sell"

    return rsi_sig, ma20_sig, ma_cross_sig, macd_sig


def weekly_prep_table(tickers, days=400, fetch_dividends=False):
    """يبني جدول تحضير أسبوعي شامل لكل الأسهم دفعة وحدة: الإغلاق الأسبوعي،
    نسبة التغيّر، إشارات قصيرة المدى، أقرب دعم/مقاومة، وتاريخ آخر توزيعة (اختياري)."""
    rows = []
    for ticker in tickers:
        df = analyze_ticker(ticker, days)
        if df is None:
            rows.append({"الرمز": ticker, "خطأ": "لا توجد بيانات"})
            continue

        weekly = df["Close"].resample("W").last().dropna()
        last_close = weekly.iloc[-1]
        prev_close = weekly.iloc[-2] if len(weekly) > 1 else last_close
        weekly_change = (last_close - prev_close) / prev_close * 100

        rsi_sig, ma20_sig, ma_cross_sig, macd_sig = short_term_signal_row(df)
        supports, resistances = support_resistance_levels(df)

        row = {
            "الرمز": ticker,
            "آخر إغلاق أسبوعي": round(float(last_close), 3),
            "تغيّر أسبوعي %": round(float(weekly_change), 2),
            "RSI": rsi_sig,
            "MA20 مقابل السعر": ma20_sig,
            "MA20 مقابل MA50": ma_cross_sig,
            "MACD": macd_sig,
            "أقرب دعم": round(supports[0], 3) if supports else None,
            "أقرب مقاومة": round(resistances[0], 3) if resistances else None,
        }

        if fetch_dividends:
            try:
                divs = yf.Ticker(ticker).dividends
                if not divs.empty:
                    row["آخر تاريخ توزيعة"] = divs.index[-1].date().isoformat()
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


def market_indices_summary(days=60):
    """يجيب أداء مؤشري دبي وأبوظبي العامين آخر أسبوع، لمعرفة هل حركة أسهمك بسبب
    السوق ككل أو بسبب السهم نفسه."""
    results = {}
    for name, ticker in MARKET_INDICES.items():
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
            if df.empty:
                results[name] = None
                continue
            weekly = df["Close"].resample("W").last().dropna()
            last_val = float(weekly.iloc[-1])
            prev_val = float(weekly.iloc[-2]) if len(weekly) > 1 else last_val
            change = (last_val - prev_val) / prev_val * 100
            results[name] = {"value": last_val, "weekly_change_pct": change}
        except Exception:
            results[name] = None
    return results


def print_backtest_results(ticker, trades, stats):
    print("\n╔══════════════ نتائج المحاكاة التاريخية (Backtest) ══════════════╗")
    if not trades or stats is None:
        print("  لا توجد صفقات كافية خلال هذه الفترة لتقييم الاستراتيجية.")
        print("╚═══════════════════════════════════════════════════════════════╝")
        return

    print(f"  عدد الصفقات:              {stats['num_trades']}")
    print(f"  نسبة الصفقات الرابحة:     {stats['win_rate']:.1f}%")
    print(f"  متوسط الربح للصفقة الرابحة: {stats['avg_win_pct']:+.2f}%")
    print(f"  متوسط الخسارة للصفقة الخاسرة: {stats['avg_loss_pct']:+.2f}%")
    print(f"  معامل الربح (Profit Factor): {stats['profit_factor']:.2f}"
          f"  (>1 يعني الأرباح أكبر من الخسائر إجمالاً)")
    print(f"  متوسط مدة الاحتفاظ بالصفقة: {stats['avg_days_held']:.1f} يوم")
    print(f"  العائد الإجمالي للاستراتيجية (تراكمي): {stats['total_return_pct']:+.1f}%")
    print(f"  عائد الشراء والاحتفاظ لنفس الفترة (Buy & Hold): {stats['buy_hold_return_pct']:+.1f}%")

    diff = stats["total_return_pct"] - stats["buy_hold_return_pct"]
    if diff > 0:
        print(f"  ✅ الاستراتيجية تفوقت على الشراء والاحتفاظ بفارق {diff:+.1f} نقطة مئوية")
    else:
        print(f"  ⚠️  الاستراتيجية كانت أضعف من الشراء والاحتفاظ بفارق {diff:+.1f} نقطة مئوية")

    print("\n  آخر 5 صفقات:")
    for tr in trades[-5:]:
        print(f"    {tr['entry_date'].date()} @ {tr['entry_price']:.2f}  ->  "
              f"{tr['exit_date'].date()} @ {tr['exit_price']:.2f}  |  "
              f"{tr['return_pct']:+.2f}%  ({tr['reason']}, {tr['days_held']} يوم)")

    print("\n  ⚠️ ملاحظات منهجية:")
    print("     - المحاكاة سببية بالكامل (لا تستخدم بيانات مستقبلية) لكنها تبقى محاكاة على بيانات")
    print("       تاريخية فقط - الأداء السابق لا يضمن أداء مستقبلي مشابه.")
    print("     - لا تشمل عمولة الوسيط ولا فرق السعر (slippage) ولا ضريبة/رسوم.")
    print("     - عدد صفقات قليل (شائع بأسهم DFM/ADX الأقل سيولة) يجعل الإحصائيات أقل موثوقية.")
    print("╚═══════════════════════════════════════════════════════════════╝")


def plot_ticker(ticker, df, outdir="/mnt/user-data/outputs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    axes[0].plot(df.index, df["Close"], label="السعر", color="black", linewidth=1.1)
    axes[0].plot(df.index, df["EMA20"], label="EMA20", linewidth=0.9)
    axes[0].plot(df.index, df["EMA50"], label="EMA50", linewidth=0.9)
    axes[0].plot(df.index, df["EMA200"], label="EMA200", linewidth=0.9)
    axes[0].plot(df.index, df["VWAP"], label="VWAP", linewidth=0.8, linestyle=":")
    axes[0].set_title(f"{ticker} - السعر والمتوسطات و VWAP")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(df.index, df["RSI14"], color="purple")
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[1].set_title("RSI (14)")

    axes[2].plot(df.index, df["MACD"], label="MACD")
    axes[2].plot(df.index, df["MACD_Signal"], label="Signal")
    axes[2].bar(df.index, df["MACD_Hist"], color="grey", alpha=0.4)
    axes[2].set_title("MACD")
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].plot(df.index, df["ADX"], label="ADX", color="black")
    axes[3].plot(df.index, df["Plus_DI"], label="+DI", color="green")
    axes[3].plot(df.index, df["Minus_DI"], label="-DI", color="red")
    axes[3].axhline(20, color="grey", linestyle="--", linewidth=0.7)
    axes[3].set_title("ADX / +DI / -DI")
    axes[3].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    path = f"{outdir}/{ticker.replace('.', '_')}_chart.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="تحليل فني وأساسي موسّع لأسهم سوق دبي وأبوظبي")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=400, help="عدد أيام البيانات (يُفضّل 400+ ليشمل EMA200)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--no-fundamentals", action="store_true", help="تخطي جلب إجماع المحللين/الأخبار (أسرع)")
    parser.add_argument("--amount", type=float, default=10000,
                         help="المبلغ المخصص للاستثمار في كل سهم بالدرهم (لحساب عدد الأسهم). الافتراضي 10000")
    parser.add_argument("--backtest", action="store_true",
                         help="تشغيل محاكاة تاريخية (Backtest) للاستراتيجية بدل/مع التحليل الحي")
    parser.add_argument("--entry-threshold", type=float, default=0.15, help="عتبة Score للدخول بصفقة (افتراضي 0.15)")
    parser.add_argument("--exit-threshold", type=float, default=-0.05, help="عتبة Score للخروج من صفقة (افتراضي -0.05)")
    parser.add_argument("--weekly-prep", action="store_true",
                         help="عرض جدول تحضير أسبوعي مختصر لكل الأسهم بدل التحليل الكامل")
    args = parser.parse_args()

    if args.weekly_prep:
        print("📅 التحضير الأسبوعي\n" + "=" * 65)
        idx = market_indices_summary()
        for name, data in idx.items():
            if data:
                print(f"{name}: {data['value']:.2f}  ({data['weekly_change_pct']:+.2f}% أسبوعياً)")
            else:
                print(f"{name}: تعذّر الجلب")
        print()
        table = weekly_prep_table(args.tickers, args.days, fetch_dividends=not args.no_fundamentals)
        print(table.to_string(index=False))
        return

    if args.backtest and args.days < 600:
        print("ℹ️  لمحاكاة تاريخية ذات معنى، تم رفع --days تلقائياً إلى 600 (كنت قد حددت أقل من ذلك).\n")
        args.days = 600

    print(f"جاري تحليل {len(args.tickers)} سهم لآخر {args.days} يوم...\n")

    for ticker in args.tickers:
        print("=" * 65)
        print(f"السهم: {ticker}")
        df = analyze_ticker(ticker, args.days)
        if df is None:
            continue

        last = df.iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100
        print(f"آخر إغلاق يومي (من بيانات التحليل): {last['Close']:.2f} ({change_pct:+.2f}%)  | الحجم: {int(last['Volume']):,}")

        try:
            t_quote = yf.Ticker(ticker)
            quote = fetch_latest_quote(t_quote)
        except Exception:
            quote = None
        if quote:
            local_time = quote["timestamp"]
            print(f"أحدث سعر متاح فعلياً: {quote['price']:.2f}  |  بتاريخ ووقت: {local_time}  |  {quote['source']}")
        else:
            print("⚠️ تعذّر جلب سعر لحظي إضافي - استُخدم آخر إغلاق يومي فقط.")

        score, verdict, notes, rr_note, extra = build_composite_signal(df, ticker)

        fund = {}
        if not args.no_fundamentals:
            try:
                t = yf.Ticker(ticker)
                fund = fetch_fundamental_info(t)
            except Exception as e:
                print(f"(تعذّر جلب البيانات الأساسية/الأخبار: {e})")

        print_decision_summary(ticker, score, verdict, extra, fund, args.amount)

        if args.backtest:
            trades = backtest_strategy(df, entry_threshold=args.entry_threshold, exit_threshold=args.exit_threshold)
            stats = summarize_backtest(trades, df)
            print_backtest_results(ticker, trades, stats)

        print("\n--- تفاصيل المؤشرات ---")
        for n in notes:
            print(f"  - {n}")

        print(f"\nنسبة المخاطرة إلى العائد: {rr_note}")

        if fund:
            print_fundamentals(fund)

        if args.csv:
            csv_path = f"/mnt/user-data/outputs/{ticker.replace('.', '_')}_data.csv"
            df.to_csv(csv_path)
            print(f"\n💾 تم حفظ البيانات: {csv_path}")

        if args.plot:
            chart_path = plot_ticker(ticker, df)
            print(f"🖼️  تم حفظ الرسم البياني: {chart_path}")

        print()

    print("=" * 65)
    print("ملاحظات مهمة:")
    print("  • سوقا دبي وأبوظبي يتداولان من الاثنين للجمعة، 10:00 ص - 3:00 م بتوقيت الإمارات (GMT+4).")
    print("    شغّل السكربت بعد إغلاق الجلسة (~3:00 م) للحصول على إغلاق اليوم نفسه بشكل مؤكد.")
    print("  • إجماع المحللين/الأخبار غير متوفرة دائماً لأسهم DFM/ADX عبر Yahoo Finance.")
    print("  • Fibonacci Bollinger Bands و Nadaraya-Watson هنا تقريبات شائعة، قد تختلف قليلاً عن TradingView.")
    print("  • هذا التحليل لأغراض تعليمية فقط وليس توصية استثمارية.")


if __name__ == "__main__":
    main()
    sys.exit(1)


DEFAULT_TICKERS = [
    "EMAAR.AE",      # إعمار العقارية - دبي
    "ETISALAT.AB",   # اتصالات e& - أبوظبي
    "ADNOCGAS.AB",   # أدنوك للغاز - أبوظبي
    "ADNOCDIST.AB",  # أدنوك للتوزيع - أبوظبي
    "ALDAR.AB",      # الدار العقارية - أبوظبي
    "EMAARDEV.AE",   # إعمار للتطوير العقاري - دبي
]

# رموز المؤشرات العامة للسوقين (لقياس هل حركة السهم بسبب السهم نفسه أو السوق كله)
MARKET_INDICES = {
    "مؤشر دبي العام (DFM General)": "DFMGI.AE",
    "مؤشر أبوظبي العام (ADX General)": "FADGI.FGI",
}


# وزن كل مؤشر (نجوم المستخدم) يُستخدم في حساب الإشارة المركّبة النهائية
WEIGHTS = {
    "nadaraya_watson": 5,
    "lin_reg_channel": 5,
    "fib_bb": 5,
    "rsi": 4,
    "macd": 4,
    "ema20": 3,
    "ema50": 4,
    "ema200": 5,
    "vwap": 4,
    "adx_di": 4,
    "mfi": 4,
    "volume": 5,
    "support_resistance": 5,
    "candlestick": 4,
    "trend_structure": 5,
}


# ============================== المؤشرات الأساسية ============================== #

def sma(series, window):
    return series.rolling(window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rma(series, period):
    """تنعيم وايلدر - يُستخدم في ATR/ADX."""
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(series, window=20, num_std=2):
    mid = sma(series, window)
    std = series.rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return rma(tr, period)


def adx_indicator(df, period=14):
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr_smoothed = atr(df, period)
    plus_di = 100 * rma(plus_dm, period) / tr_smoothed
    minus_di = 100 * rma(minus_dm, period) / tr_smoothed
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return rma(dx, period), plus_di, minus_di


def mfi_indicator(df, period=14):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical * df["Volume"]
    delta = typical.diff()
    pos_flow = money_flow.where(delta > 0, 0.0).rolling(period).sum()
    neg_flow = money_flow.where(delta < 0, 0.0).rolling(period).sum()
    mfr = pos_flow / neg_flow
    return 100 - (100 / (1 + mfr))


def vwap_cumulative(df):
    """VWAP تراكمي منذ بداية الفترة المحمّلة (تقريب مناسب للبيانات اليومية)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()


def linear_regression_channel(series, window=50, num_std=2):
    y = series.iloc[-window:].values
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    resid_std = (y - fit).std()
    return slope, fit[-1], fit[-1] + num_std * resid_std, fit[-1] - num_std * resid_std


def nadaraya_watson_envelope(series, window=50, bandwidth=8, mult=3):
    """نسخة غير مُعاد رسمها (non-repainting): كل نقطة تُحسب فقط من البيانات
    السابقة لها (نافذة سببية causal)، بدون استخدام بيانات مستقبلية."""
    values = series.values
    n = len(values)
    est = np.full(n, np.nan)
    kernel_idx = np.arange(window)
    weights = np.exp(-((window - 1 - kernel_idx) ** 2) / (2 * bandwidth ** 2))
    for i in range(window, n):
        est[i] = np.sum(weights * values[i - window:i]) / weights.sum()
    est_series = pd.Series(est, index=series.index)
    mae = (series - est_series).abs().rolling(window).mean()
    return est_series, est_series + mae * mult, est_series - mae * mult


def fibonacci_bollinger_bands(series, window=20):
    """تقريب شائع لـ Fibonacci Bollinger Bands: خط أساس SMA، ونطاقات
    بمضاعفات فيبوناتشي لانحراف معياري مضاعف بدل مضاعف بولينجر الثابت (2)."""
    basis = sma(series, window)
    dev = series.rolling(window).std() * 3
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
    upper = {f: basis + dev * f for f in fib_ratios}
    lower = {f: basis - dev * f for f in fib_ratios}
    return basis, upper, lower


def rolling_linreg(series, window=50, num_std=2):
    """نسخة (rolling) من قناة الانحدار الخطي تُرجع مصفوفة لكل يوم، بحيث كل
    نقطة تعتمد فقط على النافذة السابقة لها مباشرة (سببية بالكامل - تصلح للمحاكاة التاريخية)."""
    values = series.values
    n = len(values)
    slope_arr = np.full(n, np.nan)
    fit_arr = np.full(n, np.nan)
    upper_arr = np.full(n, np.nan)
    lower_arr = np.full(n, np.nan)
    x = np.arange(window)
    for i in range(window, n):
        y = values[i - window:i]
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * x + intercept
        resid_std = (y - fit).std()
        slope_arr[i] = slope
        fit_arr[i] = fit[-1]
        upper_arr[i] = fit[-1] + num_std * resid_std
        lower_arr[i] = fit[-1] - num_std * resid_std
    return slope_arr, fit_arr, upper_arr, lower_arr


def find_swing_points_positions(df, order=5):
    """مثل find_swing_points لكن تُرجع رقم الموضع (index position) بدل التاريخ،
    لتسهيل التحقق من "متى صارت القمة/القاع مؤكدة" أثناء المحاكاة التاريخية."""
    highs, lows = df["High"].values, df["Low"].values
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(order, n - order):
        if highs[i] == highs[i - order:i + order + 1].max():
            swing_highs.append((i, highs[i]))
        if lows[i] == lows[i - order:i + order + 1].min():
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def candlestick_signal(o, h, l, c, po, pc):
    """نسخة رقمية سريعة من detect_candlestick لاستخدامها داخل حلقة المحاكاة التاريخية."""
    body = abs(c - o)
    rng = h - l if h != l else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body / rng < 0.1:
        return 0
    if c > o and c > po and o < pc and body > abs(pc - po):
        return 1
    if c < o and c < po and o > pc and body > abs(pc - po):
        return -1
    if lower_wick > 2 * body and upper_wick < body:
        return 1
    if upper_wick > 2 * body and lower_wick < body:
        return -1
    return 0


# ============================== أدوات هيكلية ============================== #

def find_swing_points(df, order=5):
    highs, lows = df["High"], df["Low"]
    n = len(df)
    swing_highs, swing_lows = [], []
    for i in range(order, n - order):
        wh = highs.iloc[i - order:i + order + 1]
        wl = lows.iloc[i - order:i + order + 1]
        if highs.iloc[i] == wh.max():
            swing_highs.append((df.index[i], highs.iloc[i]))
        if lows.iloc[i] == wl.min():
            swing_lows.append((df.index[i], lows.iloc[i]))
    return swing_highs, swing_lows


def support_resistance_levels(df, order=5, n_levels=3):
    swing_highs, swing_lows = find_swing_points(df, order)
    price = df["Close"].iloc[-1]
    resistances = sorted([lvl for _, lvl in swing_highs if lvl > price])[:n_levels]
    supports = sorted([lvl for _, lvl in swing_lows if lvl < price], reverse=True)[:n_levels]
    return supports, resistances


def trend_structure(df, order=5, n_swings=4):
    swing_highs, swing_lows = find_swing_points(df, order)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "غير كافٍ لتحديد الهيكل", 0
    last_highs = [h for _, h in swing_highs[-n_swings:]]
    last_lows = [l for _, l in swing_lows[-n_swings:]]
    higher_highs = all(x < y for x, y in zip(last_highs, last_highs[1:]))
    higher_lows = all(x < y for x, y in zip(last_lows, last_lows[1:]))
    lower_highs = all(x > y for x, y in zip(last_highs, last_highs[1:]))
    lower_lows = all(x > y for x, y in zip(last_lows, last_lows[1:]))

    if higher_highs and higher_lows:
        return "اتجاه صاعد واضح (Higher Highs / Higher Lows)", 1
    if lower_highs and lower_lows:
        return "اتجاه هابط واضح (Lower Highs / Lower Lows)", -1
    return "هيكل متذبذب / غير محدد الاتجاه", 0


def detect_candlestick(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = last["Open"], last["High"], last["Low"], last["Close"]
    po, pc = prev["Open"], prev["Close"]
    body = abs(c - o)
    rng = h - l if h != l else 1e-9
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if body / rng < 0.1:
        return "دوجي (Doji) - تردد في السوق", 0
    if c > o and c > po and o < pc and body > abs(pc - po):
        return "ابتلاع صاعد (Bullish Engulfing)", 1
    if c < o and c < po and o > pc and body > abs(pc - po):
        return "ابتلاع هابط (Bearish Engulfing)", -1
    if lower_wick > 2 * body and upper_wick < body:
        return "مطرقة (Hammer) - احتمال ارتداد صاعد", 1
    if upper_wick > 2 * body and lower_wick < body:
        return "نجمة الرماية (Shooting Star) - احتمال ارتداد هابط", -1
    return "شمعة عادية بدون نمط واضح", 0


# ============================== بيانات إضافية (اختيارية) ============================== #

def fetch_fundamental_info(ticker_obj):
    """يحاول جلب إجماع المحللين، الأرباح، التوزيعات، والأخبار. بعض هذه
    البيانات قد لا تتوفر لأسهم سوق الإمارات على Yahoo Finance."""
    result = {}
    try:
        info = ticker_obj.get_info()
    except Exception:
        info = {}
    result["recommendation"] = info.get("recommendationKey")
    result["target_mean"] = info.get("targetMeanPrice")
    result["target_high"] = info.get("targetHighPrice")
    result["target_low"] = info.get("targetLowPrice")
    result["num_analysts"] = info.get("numberOfAnalystOpinions")
    result["dividend_yield"] = info.get("dividendYield")

    try:
        divs = ticker_obj.dividends
        result["last_dividends"] = divs.tail(3) if not divs.empty else None
    except Exception:
        result["last_dividends"] = None

    try:
        earnings_dates = ticker_obj.get_earnings_dates(limit=4)
        result["earnings_dates"] = earnings_dates
    except Exception:
        result["earnings_dates"] = None

    try:
        news = ticker_obj.news
        result["news"] = news[:3] if news else []
    except Exception:
        result["news"] = []

    return result


# ============================== التحليل الرئيسي ============================== #

def fetch_latest_quote(ticker_obj):
    """يحاول جلب أحدث نقطة سعر متاحة فعلياً (بيانات دقيقة إن وُجدت لهذا السهم
    على Yahoo، وإلا آخر إغلاق يومي)، مع طابعها الزمني الحقيقي حتى يتضح للمستخدم
    بوضوح هل هذا فعلاً "سعر اليوم" أم آخر جلسة تداول سابقة."""
    try:
        intraday = ticker_obj.history(period="1d", interval="1m")
    except Exception:
        intraday = None

    if intraday is not None and not intraday.empty:
        return {
            "price": intraday["Close"].iloc[-1],
            "timestamp": intraday.index[-1],
            "source": "بيانات دقيقة (قريبة من اللحظي، عادة بتأخير 15-20 دقيقة)",
        }

    try:
        daily = ticker_obj.history(period="5d")
    except Exception:
        daily = None

    if daily is not None and not daily.empty:
        return {
            "price": daily["Close"].iloc[-1],
            "timestamp": daily.index[-1],
            "source": "آخر إغلاق يومي متاح (لا توجد بيانات داخل اليوم لهذا السهم على Yahoo)",
        }

    return None


def analyze_ticker(ticker, days):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        print(f"⚠️  لا توجد بيانات للرمز: {ticker}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    df["EMA20"] = ema(close, 20)
    df["EMA50"] = ema(close, 50)
    df["EMA200"] = ema(close, 200)
    df["SMA20"] = sma(close, 20)
    df["RSI14"] = rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = bollinger_bands(close)
    df["ATR14"] = atr(df)
    df["ADX"], df["Plus_DI"], df["Minus_DI"] = adx_indicator(df)
    df["MFI14"] = mfi_indicator(df)
    df["VWAP"] = vwap_cumulative(df)
    df["Vol_Avg20"] = df["Volume"].rolling(20).mean()

    return df


def build_composite_signal(df, ticker):
    last = df.iloc[-1]
    price = last["Close"]
    signals = {}
    notes = []

    signals["ema20"] = 1 if price > last["EMA20"] else -1
    signals["ema50"] = 1 if price > last["EMA50"] else -1
    signals["ema200"] = 1 if price > last["EMA200"] else -1
    notes.append(f"EMA20/50/200: {last['EMA20']:.2f} / {last['EMA50']:.2f} / {last['EMA200']:.2f}")

    if last["RSI14"] < 30:
        signals["rsi"] = 1
    elif last["RSI14"] > 70:
        signals["rsi"] = -1
    else:
        signals["rsi"] = 0
    notes.append(f"RSI(14): {last['RSI14']:.1f}")

    signals["macd"] = 1 if last["MACD"] > last["MACD_Signal"] else -1
    notes.append(f"MACD: {last['MACD']:.4f} مقابل الإشارة {last['MACD_Signal']:.4f}")

    if last["ADX"] > 20:
        signals["adx_di"] = 1 if last["Plus_DI"] > last["Minus_DI"] else -1
    else:
        signals["adx_di"] = 0
    notes.append(f"ADX: {last['ADX']:.1f} (قوة الاتجاه) | +DI {last['Plus_DI']:.1f} / -DI {last['Minus_DI']:.1f}")

    if last["MFI14"] < 20:
        signals["mfi"] = 1
    elif last["MFI14"] > 80:
        signals["mfi"] = -1
    else:
        signals["mfi"] = 0
    notes.append(f"MFI(14): {last['MFI14']:.1f}")

    vol_ratio = last["Volume"] / last["Vol_Avg20"] if last["Vol_Avg20"] > 0 else 1
    price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
    if vol_ratio > 1.2:
        signals["volume"] = 1 if price_change > 0 else -1
    else:
        signals["volume"] = 0
    notes.append(f"الحجم مقارنة بمتوسط 20 يوم: {vol_ratio:.2f}x")

    signals["vwap"] = 1 if price > last["VWAP"] else -1
    notes.append(f"VWAP: {last['VWAP']:.2f}")

    supports, resistances = support_resistance_levels(df)
    if supports and abs(price - supports[0]) / price < 0.015:
        signals["support_resistance"] = 1
        sr_note = f"السعر قريب من دعم عند {supports[0]:.2f}"
    elif resistances and abs(price - resistances[0]) / price < 0.015:
        signals["support_resistance"] = -1
        sr_note = f"السعر قريب من مقاومة عند {resistances[0]:.2f}"
    else:
        signals["support_resistance"] = 0
        sr_note = "السعر بين الدعم والمقاومة الحاليين"
    notes.append(sr_note)
    notes.append(f"أقرب دعوم: {[round(s,2) for s in supports]} | أقرب مقاومات: {[round(r,2) for r in resistances]}")

    candle_desc, candle_signal = detect_candlestick(df)
    signals["candlestick"] = candle_signal
    notes.append(f"الشمعة الأخيرة: {candle_desc}")

    structure_desc, structure_signal = trend_structure(df)
    signals["trend_structure"] = structure_signal
    notes.append(f"هيكل الاتجاه: {structure_desc}")

    slope, lr_value, lr_upper, lr_lower = linear_regression_channel(df["Close"])
    signals["lin_reg_channel"] = 1 if slope > 0 else -1
    notes.append(f"قناة الانحدار الخطي: الميل {'صاعد' if slope>0 else 'هابط'}, القناة [{lr_lower:.2f} - {lr_upper:.2f}]")

    nw_est, nw_upper, nw_lower = nadaraya_watson_envelope(df["Close"])
    if not np.isnan(nw_upper.iloc[-1]):
        if price < nw_lower.iloc[-1]:
            signals["nadaraya_watson"] = 1
        elif price > nw_upper.iloc[-1]:
            signals["nadaraya_watson"] = -1
        else:
            signals["nadaraya_watson"] = 0
        notes.append(f"Nadaraya-Watson: القيمة المقدرة {nw_est.iloc[-1]:.2f}, النطاق [{nw_lower.iloc[-1]:.2f} - {nw_upper.iloc[-1]:.2f}]")
    else:
        signals["nadaraya_watson"] = 0
        notes.append("Nadaraya-Watson: بيانات غير كافية بعد")

    fib_basis, fib_upper, fib_lower = fibonacci_bollinger_bands(df["Close"])
    fu_1 = fib_upper[1.0].iloc[-1]
    fl_1 = fib_lower[1.0].iloc[-1]
    if not np.isnan(fu_1):
        if price <= fib_lower[0.382].iloc[-1]:
            signals["fib_bb"] = 1
        elif price >= fib_upper[0.382].iloc[-1]:
            signals["fib_bb"] = -1
        else:
            signals["fib_bb"] = 0
        notes.append(f"Fibonacci Bollinger Bands: أساس {fib_basis.iloc[-1]:.2f}, نطاق كامل [{fl_1:.2f} - {fu_1:.2f}]")
    else:
        signals["fib_bb"] = 0

    notes.append(f"ATR(14): {last['ATR14']:.3f} (مقياس التذبذب - يُستخدم لتحديد حجم وقف الخسارة)")

    total_weight = sum(WEIGHTS[k] for k in signals)
    weighted_sum = sum(WEIGHTS[k] * v for k, v in signals.items())
    score = weighted_sum / total_weight  # بين -1 و 1

    if score >= 0.35:
        verdict = "📈 إشارة إيجابية قوية"
    elif score >= 0.1:
        verdict = "📈 إشارة إيجابية"
    elif score <= -0.35:
        verdict = "📉 إشارة سلبية قوية"
    elif score <= -0.1:
        verdict = "📉 إشارة سلبية"
    else:
        verdict = "➖ محايدة"

    # نسبة المخاطرة إلى العائد بناءً على أقرب دعم/مقاومة
    rr_note = "غير متاحة (لا يوجد دعم أو مقاومة واضحة قريبة)"
    if supports and resistances:
        risk = price - supports[0]
        reward = resistances[0] - price
        if risk > 0:
            rr = reward / risk
            rr_note = f"1 : {rr:.2f}  (وقف مقترح {supports[0]:.2f} | هدف مقترح {resistances[0]:.2f})"

    extra = {
        "price": price,
        "supports": supports,
        "resistances": resistances,
        "nw_lower": nw_lower.iloc[-1] if not np.isnan(nw_upper.iloc[-1]) else None,
        "nw_upper": nw_upper.iloc[-1] if not np.isnan(nw_upper.iloc[-1]) else None,
    }
    return score, verdict, notes, rr_note, extra


def print_fundamentals(fund):
    print("\n--- إجماع المحللين والبيانات الأساسية ---")
    if fund["recommendation"]:
        print(f"توصية المحللين: {fund['recommendation']}")
    if fund["target_mean"]:
        print(f"متوسط السعر المستهدف: {fund['target_mean']:.2f} "
              f"(نطاق {fund['target_low']:.2f} - {fund['target_high']:.2f}, عدد المحللين: {fund['num_analysts']})")
    if not fund["recommendation"] and not fund["target_mean"]:
        print("لا تتوفر بيانات تحليل محللين لهذا السهم على Yahoo Finance حالياً.")

    if fund["dividend_yield"]:
        print(f"عائد التوزيعات: {fund['dividend_yield']*100:.2f}%")
    if fund["last_dividends"] is not None and len(fund["last_dividends"]) > 0:
        print("آخر توزيعات:")
        for date, val in fund["last_dividends"].items():
            print(f"  {date.date()}: {val:.3f}")

    if fund["earnings_dates"] is not None and len(fund["earnings_dates"]) > 0:
        print("مواعيد الأرباح القادمة/الأخيرة:")
        print(fund["earnings_dates"].head(3).to_string())

    if fund["news"]:
        print("آخر الأخبار (العناوين فقط - راجع الرابط للتفاصيل):")
        for item in fund["news"]:
            title = item.get("title") or item.get("content", {}).get("title")
            link = item.get("link") or item.get("content", {}).get("clickThroughUrl", {}).get("url")
            if title:
                print(f"  • {title}")
                if link:
                    print(f"    {link}")
    else:
        print("لا توجد أخبار حديثة متاحة عبر الـ API لهذا السهم.")


def decision_from_score(score):
    if score >= 0.35:
        return "شراء قوي"
    elif score >= 0.1:
        return "شراء تدريجي"
    elif score <= -0.35:
        return "تجنب / بيع"
    elif score <= -0.1:
        return "تقليل / حذر"
    return "انتظار ومراقبة"


def print_decision_summary(ticker, score, verdict, extra, fund, amount):
    price = extra["price"]
    supports = extra["supports"]
    resistances = extra["resistances"]

    # منطقة الشراء المقترحة: بين أقرب دعم (أو النطاق السفلي لـ Nadaraya-Watson إن كان أقرب) والسعر الحالي
    zone_low_candidates = [v for v in [supports[0] if supports else None, extra["nw_lower"]] if v is not None]
    zone_low = max(zone_low_candidates) if zone_low_candidates else price * 0.97
    buy_zone = f"{zone_low:.2f} - {price:.2f}"

    # السعر المستهدف (12 شهر): تقديرات المحللين إن توفرت، وإلا أقرب مقاومة بعيدة كتقدير فني تقريبي
    if fund.get("target_mean"):
        target = fund["target_mean"]
        target_source = f"(إجماع {fund.get('num_analysts') or '؟'} محلل)"
    elif resistances:
        target = resistances[-1]
        target_source = "(تقدير فني تقريبي من أبعد مقاومة، وليس تقديراً مالياً لـ 12 شهر)"
    else:
        target = price * 1.10
        target_source = "(تقدير عام تقريبي - لا تتوفر بيانات كافية)"
    upside_pct = (target - price) / price * 100

    shares = int(amount // price) if price > 0 else 0
    actual_cost = shares * price
    leftover = amount - actual_cost

    recommendation = fund.get("recommendation") or "غير متوفرة"

    print("\n╔══════════════ ملخص القرار ══════════════╗")
    print(f"  السعر الحالي:            {price:.2f} AED")
    print(f"  إجماع المحللين:          {recommendation}")
    print(f"  درجة الإشارة (Score):    {score:+.2f}  -  {verdict}")
    print(f"  القرار:                  {decision_from_score(score)}")
    print(f"  منطقة الشراء المقترحة:   {buy_zone} AED")
    print(f"  السعر المستهدف (12 شهر): {target:.2f} AED  {target_source}")
    print(f"  نسبة الصعود المحتملة:    {upside_pct:+.1f}%")
    print(f"  المبلغ المخصص:           {amount:,.0f} AED")
    print(f"  عدد الأسهم الممكن شراؤها: {shares:,}  (تكلفة فعلية {actual_cost:,.0f} AED، متبقي {leftover:,.0f} AED)")

    news = fund.get("news") or []
    print("  أهم الأخبار:")
    if news:
        for item in news[:2]:
            title = item.get("title") or item.get("content", {}).get("title")
            if title:
                print(f"    • {title}")
    else:
        print("    لا تتوفر أخبار حديثة عبر الـ API لهذا السهم.")
    print("╚═══════════════════════════════════════════╝")


def backtest_strategy(df, order=5, entry_threshold=0.15, exit_threshold=-0.05,
                       stop_atr_mult=2.0, target_atr_mult=4.0):
    """
    محاكاة تاريخية لاستراتيجية Long-only بسيطة مبنية على نفس الإشارة المركّبة
    المستخدمة في التحليل الحي، لكن بحساب كل مؤشر بشكل سببي بحت (كل يوم يعتمد
    فقط على بيانات الماضي حتى ذلك اليوم) لتفادي Look-ahead Bias.

    قواعد الصفقة:
      - دخول: أول يوم يتجاوز فيه Score قيمة entry_threshold وما فيه صفقة مفتوحة.
      - خروج: أول ما يحصل من: وقف خسارة (سعر الدخول - ATR×stop_atr_mult)،
        أو هدف ربح (سعر الدخول + ATR×target_atr_mult)، أو انعكاس الإشارة تحت exit_threshold.
      - بدون عمولة أو انزلاق سعري (slippage) - النتائج الفعلية غالباً أضعف قليلاً من المحاكاة.
    """
    n = len(df)
    close, open_ = df["Close"].values, df["Open"].values
    high, low = df["High"].values, df["Low"].values

    ema20, ema50, ema200 = df["EMA20"].values, df["EMA50"].values, df["EMA200"].values
    rsi14 = df["RSI14"].values
    macd_l, macd_s = df["MACD"].values, df["MACD_Signal"].values
    adx_v, plus_di, minus_di = df["ADX"].values, df["Plus_DI"].values, df["Minus_DI"].values
    mfi14 = df["MFI14"].values
    vwap = df["VWAP"].values
    vol, vol_avg20 = df["Volume"].values, df["Vol_Avg20"].values
    atr14 = df["ATR14"].values

    _, nw_upper_s, nw_lower_s = nadaraya_watson_envelope(df["Close"])
    nw_upper, nw_lower = nw_upper_s.values, nw_lower_s.values

    _, fib_upper_d, fib_lower_d = fibonacci_bollinger_bands(df["Close"])
    fib_u38, fib_l38 = fib_upper_d[0.382].values, fib_lower_d[0.382].values

    slope_arr, _, _, _ = rolling_linreg(df["Close"])
    swing_highs_pos, swing_lows_pos = find_swing_points_positions(df, order)

    trades = []
    position = None
    start_idx = 200  # حتى يكتمل EMA200

    for t in range(start_idx, n):
        price = close[t]

        confirmed_highs = [lvl for i, lvl in swing_highs_pos if i + order <= t and lvl > price]
        confirmed_lows = [lvl for i, lvl in swing_lows_pos if i + order <= t and lvl < price]
        resistances = sorted(confirmed_highs)[:3]
        supports = sorted(confirmed_lows, reverse=True)[:3]

        signals = {
            "ema20": 1 if price > ema20[t] else -1,
            "ema50": 1 if price > ema50[t] else -1,
            "ema200": 1 if price > ema200[t] else -1,
            "rsi": 1 if rsi14[t] < 30 else (-1 if rsi14[t] > 70 else 0),
            "macd": 1 if macd_l[t] > macd_s[t] else -1,
            "adx_di": (1 if plus_di[t] > minus_di[t] else -1) if adx_v[t] > 20 else 0,
            "mfi": 1 if mfi14[t] < 20 else (-1 if mfi14[t] > 80 else 0),
            "vwap": 1 if price > vwap[t] else -1,
        }

        vol_ratio = vol[t] / vol_avg20[t] if vol_avg20[t] > 0 else 1
        price_chg = close[t] - close[t - 1]
        signals["volume"] = (1 if price_chg > 0 else -1) if vol_ratio > 1.2 else 0

        if supports and abs(price - supports[0]) / price < 0.015:
            signals["support_resistance"] = 1
        elif resistances and abs(price - resistances[0]) / price < 0.015:
            signals["support_resistance"] = -1
        else:
            signals["support_resistance"] = 0

        signals["candlestick"] = candlestick_signal(open_[t], high[t], low[t], close[t], open_[t - 1], close[t - 1])

        conf_h = [(i, lvl) for i, lvl in swing_highs_pos if i + order <= t]
        conf_l = [(i, lvl) for i, lvl in swing_lows_pos if i + order <= t]
        if len(conf_h) >= 2 and len(conf_l) >= 2:
            hh, hl = conf_h[-1][1] > conf_h[-2][1], conf_l[-1][1] > conf_l[-2][1]
            lh, ll = conf_h[-1][1] < conf_h[-2][1], conf_l[-1][1] < conf_l[-2][1]
            signals["trend_structure"] = 1 if (hh and hl) else (-1 if (lh and ll) else 0)
        else:
            signals["trend_structure"] = 0

        signals["lin_reg_channel"] = (1 if slope_arr[t] > 0 else -1) if not np.isnan(slope_arr[t]) else 0
        signals["nadaraya_watson"] = (1 if price < nw_lower[t] else (-1 if price > nw_upper[t] else 0)) if not np.isnan(nw_upper[t]) else 0
        signals["fib_bb"] = (1 if price <= fib_l38[t] else (-1 if price >= fib_u38[t] else 0)) if not np.isnan(fib_u38[t]) else 0

        total_w = sum(WEIGHTS[k] for k in signals)
        score = sum(WEIGHTS[k] * v for k, v in signals.items()) / total_w

        if position is None:
            if score >= entry_threshold:
                position = {
                    "entry_idx": t, "entry_price": price,
                    "stop": price - atr14[t] * stop_atr_mult,
                    "target": price + atr14[t] * target_atr_mult,
                }
        else:
            exit_price, exit_reason = None, None
            if low[t] <= position["stop"]:
                exit_price, exit_reason = position["stop"], "وقف خسارة"
            elif high[t] >= position["target"]:
                exit_price, exit_reason = position["target"], "هدف ربح"
            elif score <= exit_threshold:
                exit_price, exit_reason = price, "الإشارة انعكست"

            if exit_reason:
                ret_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
                trades.append({
                    "entry_date": df.index[position["entry_idx"]], "entry_price": position["entry_price"],
                    "exit_date": df.index[t], "exit_price": exit_price,
                    "return_pct": ret_pct, "reason": exit_reason,
                    "days_held": t - position["entry_idx"],
                })
                position = None

    if position is not None:
        t = n - 1
        ret_pct = (close[t] - position["entry_price"]) / position["entry_price"] * 100
        trades.append({
            "entry_date": df.index[position["entry_idx"]], "entry_price": position["entry_price"],
            "exit_date": df.index[t], "exit_price": close[t],
            "return_pct": ret_pct, "reason": "نهاية الفترة (صفقة مفتوحة)",
            "days_held": t - position["entry_idx"],
        })

    return trades


def summarize_backtest(trades, df, start_idx=200):
    if not trades:
        return None
    returns = [tr["return_pct"] for tr in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    compounded = 1.0
    for r in returns:
        compounded *= (1 + r / 100)

    bh_start = df["Close"].iloc[start_idx]
    bh_end = df["Close"].iloc[-1]
    bh_return = (bh_end - bh_start) / bh_start * 100

    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(returns) * 100,
        "avg_win_pct": np.mean(wins) if wins else 0,
        "avg_loss_pct": np.mean(losses) if losses else 0,
        "total_return_pct": (compounded - 1) * 100,
        "buy_hold_return_pct": bh_return,
        "avg_days_held": np.mean([tr["days_held"] for tr in trades]),
        "profit_factor": profit_factor,
    }


def short_term_signal_row(df):
    """يبني إشارات قصيرة المدى بنفس أسلوب تطبيقات الوسطاء (Buy/Sell/Hold) لكل مؤشر."""
    last = df.iloc[-1]

    if last["RSI14"] < 30:
        rsi_sig = "Buy"
    elif last["RSI14"] > 70:
        rsi_sig = "Sell"
    else:
        rsi_sig = "Hold"

    ma20_sig = "Buy" if last["Close"] > last["EMA20"] else "Sell"
    ma_cross_sig = "Buy" if last["EMA20"] > last["EMA50"] else "Sell"
    macd_sig = "Buy" if last["MACD"] > last["MACD_Signal"] else "Sell"

    return rsi_sig, ma20_sig, ma_cross_sig, macd_sig


def weekly_prep_table(tickers, days=400, fetch_dividends=False):
    """يبني جدول تحضير أسبوعي شامل لكل الأسهم دفعة وحدة: الإغلاق الأسبوعي،
    نسبة التغيّر، إشارات قصيرة المدى، أقرب دعم/مقاومة، وتاريخ آخر توزيعة (اختياري)."""
    rows = []
    for ticker in tickers:
        df = analyze_ticker(ticker, days)
        if df is None:
            rows.append({"الرمز": ticker, "خطأ": "لا توجد بيانات"})
            continue

        weekly = df["Close"].resample("W").last().dropna()
        last_close = weekly.iloc[-1]
        prev_close = weekly.iloc[-2] if len(weekly) > 1 else last_close
        weekly_change = (last_close - prev_close) / prev_close * 100

        rsi_sig, ma20_sig, ma_cross_sig, macd_sig = short_term_signal_row(df)
        supports, resistances = support_resistance_levels(df)

        row = {
            "الرمز": ticker,
            "آخر إغلاق أسبوعي": round(float(last_close), 3),
            "تغيّر أسبوعي %": round(float(weekly_change), 2),
            "RSI": rsi_sig,
            "MA20 مقابل السعر": ma20_sig,
            "MA20 مقابل MA50": ma_cross_sig,
            "MACD": macd_sig,
            "أقرب دعم": round(supports[0], 3) if supports else None,
            "أقرب مقاومة": round(resistances[0], 3) if resistances else None,
        }

        if fetch_dividends:
            try:
                divs = yf.Ticker(ticker).dividends
                if not divs.empty:
                    row["آخر تاريخ توزيعة"] = divs.index[-1].date().isoformat()
            except Exception:
                pass

        rows.append(row)

    return pd.DataFrame(rows)


def market_indices_summary(days=60):
    """يجيب أداء مؤشري دبي وأبوظبي العامين آخر أسبوع، لمعرفة هل حركة أسهمك بسبب
    السوق ككل أو بسبب السهم نفسه."""
    results = {}
    for name, ticker in MARKET_INDICES.items():
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
            if df.empty:
                results[name] = None
                continue
            weekly = df["Close"].resample("W").last().dropna()
            last_val = float(weekly.iloc[-1])
            prev_val = float(weekly.iloc[-2]) if len(weekly) > 1 else last_val
            change = (last_val - prev_val) / prev_val * 100
            results[name] = {"value": last_val, "weekly_change_pct": change}
        except Exception:
            results[name] = None
    return results


def print_backtest_results(ticker, trades, stats):
    print("\n╔══════════════ نتائج المحاكاة التاريخية (Backtest) ══════════════╗")
    if not trades or stats is None:
        print("  لا توجد صفقات كافية خلال هذه الفترة لتقييم الاستراتيجية.")
        print("╚═══════════════════════════════════════════════════════════════╝")
        return

    print(f"  عدد الصفقات:              {stats['num_trades']}")
    print(f"  نسبة الصفقات الرابحة:     {stats['win_rate']:.1f}%")
    print(f"  متوسط الربح للصفقة الرابحة: {stats['avg_win_pct']:+.2f}%")
    print(f"  متوسط الخسارة للصفقة الخاسرة: {stats['avg_loss_pct']:+.2f}%")
    print(f"  معامل الربح (Profit Factor): {stats['profit_factor']:.2f}"
          f"  (>1 يعني الأرباح أكبر من الخسائر إجمالاً)")
    print(f"  متوسط مدة الاحتفاظ بالصفقة: {stats['avg_days_held']:.1f} يوم")
    print(f"  العائد الإجمالي للاستراتيجية (تراكمي): {stats['total_return_pct']:+.1f}%")
    print(f"  عائد الشراء والاحتفاظ لنفس الفترة (Buy & Hold): {stats['buy_hold_return_pct']:+.1f}%")

    diff = stats["total_return_pct"] - stats["buy_hold_return_pct"]
    if diff > 0:
        print(f"  ✅ الاستراتيجية تفوقت على الشراء والاحتفاظ بفارق {diff:+.1f} نقطة مئوية")
    else:
        print(f"  ⚠️  الاستراتيجية كانت أضعف من الشراء والاحتفاظ بفارق {diff:+.1f} نقطة مئوية")

    print("\n  آخر 5 صفقات:")
    for tr in trades[-5:]:
        print(f"    {tr['entry_date'].date()} @ {tr['entry_price']:.2f}  ->  "
              f"{tr['exit_date'].date()} @ {tr['exit_price']:.2f}  |  "
              f"{tr['return_pct']:+.2f}%  ({tr['reason']}, {tr['days_held']} يوم)")

    print("\n  ⚠️ ملاحظات منهجية:")
    print("     - المحاكاة سببية بالكامل (لا تستخدم بيانات مستقبلية) لكنها تبقى محاكاة على بيانات")
    print("       تاريخية فقط - الأداء السابق لا يضمن أداء مستقبلي مشابه.")
    print("     - لا تشمل عمولة الوسيط ولا فرق السعر (slippage) ولا ضريبة/رسوم.")
    print("     - عدد صفقات قليل (شائع بأسهم DFM/ADX الأقل سيولة) يجعل الإحصائيات أقل موثوقية.")
    print("╚═══════════════════════════════════════════════════════════════╝")


def plot_ticker(ticker, df, outdir="/mnt/user-data/outputs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    axes[0].plot(df.index, df["Close"], label="السعر", color="black", linewidth=1.1)
    axes[0].plot(df.index, df["EMA20"], label="EMA20", linewidth=0.9)
    axes[0].plot(df.index, df["EMA50"], label="EMA50", linewidth=0.9)
    axes[0].plot(df.index, df["EMA200"], label="EMA200", linewidth=0.9)
    axes[0].plot(df.index, df["VWAP"], label="VWAP", linewidth=0.8, linestyle=":")
    axes[0].set_title(f"{ticker} - السعر والمتوسطات و VWAP")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(df.index, df["RSI14"], color="purple")
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[1].set_title("RSI (14)")

    axes[2].plot(df.index, df["MACD"], label="MACD")
    axes[2].plot(df.index, df["MACD_Signal"], label="Signal")
    axes[2].bar(df.index, df["MACD_Hist"], color="grey", alpha=0.4)
    axes[2].set_title("MACD")
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].plot(df.index, df["ADX"], label="ADX", color="black")
    axes[3].plot(df.index, df["Plus_DI"], label="+DI", color="green")
    axes[3].plot(df.index, df["Minus_DI"], label="-DI", color="red")
    axes[3].axhline(20, color="grey", linestyle="--", linewidth=0.7)
    axes[3].set_title("ADX / +DI / -DI")
    axes[3].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    path = f"{outdir}/{ticker.replace('.', '_')}_chart.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="تحليل فني وأساسي موسّع لأسهم سوق دبي وأبوظبي")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=400, help="عدد أيام البيانات (يُفضّل 400+ ليشمل EMA200)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--no-fundamentals", action="store_true", help="تخطي جلب إجماع المحللين/الأخبار (أسرع)")
    parser.add_argument("--amount", type=float, default=10000,
                         help="المبلغ المخصص للاستثمار في كل سهم بالدرهم (لحساب عدد الأسهم). الافتراضي 10000")
    parser.add_argument("--backtest", action="store_true",
                         help="تشغيل محاكاة تاريخية (Backtest) للاستراتيجية بدل/مع التحليل الحي")
    parser.add_argument("--entry-threshold", type=float, default=0.15, help="عتبة Score للدخول بصفقة (افتراضي 0.15)")
    parser.add_argument("--exit-threshold", type=float, default=-0.05, help="عتبة Score للخروج من صفقة (افتراضي -0.05)")
    parser.add_argument("--weekly-prep", action="store_true",
                         help="عرض جدول تحضير أسبوعي مختصر لكل الأسهم بدل التحليل الكامل")
    args = parser.parse_args()

    if args.weekly_prep:
        print("📅 التحضير الأسبوعي\n" + "=" * 65)
        idx = market_indices_summary()
        for name, data in idx.items():
            if data:
                print(f"{name}: {data['value']:.2f}  ({data['weekly_change_pct']:+.2f}% أسبوعياً)")
            else:
                print(f"{name}: تعذّر الجلب")
        print()
        table = weekly_prep_table(args.tickers, args.days, fetch_dividends=not args.no_fundamentals)
        print(table.to_string(index=False))
        return

    if args.backtest and args.days < 600:
        print("ℹ️  لمحاكاة تاريخية ذات معنى، تم رفع --days تلقائياً إلى 600 (كنت قد حددت أقل من ذلك).\n")
        args.days = 600

    print(f"جاري تحليل {len(args.tickers)} سهم لآخر {args.days} يوم...\n")

    for ticker in args.tickers:
        print("=" * 65)
        print(f"السهم: {ticker}")
        df = analyze_ticker(ticker, args.days)
        if df is None:
            continue

        last = df.iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100
        print(f"آخر إغلاق يومي (من بيانات التحليل): {last['Close']:.2f} ({change_pct:+.2f}%)  | الحجم: {int(last['Volume']):,}")

        try:
            t_quote = yf.Ticker(ticker)
            quote = fetch_latest_quote(t_quote)
        except Exception:
            quote = None
        if quote:
            local_time = quote["timestamp"]
            print(f"أحدث سعر متاح فعلياً: {quote['price']:.2f}  |  بتاريخ ووقت: {local_time}  |  {quote['source']}")
        else:
            print("⚠️ تعذّر جلب سعر لحظي إضافي - استُخدم آخر إغلاق يومي فقط.")

        score, verdict, notes, rr_note, extra = build_composite_signal(df, ticker)

        fund = {}
        if not args.no_fundamentals:
            try:
                t = yf.Ticker(ticker)
                fund = fetch_fundamental_info(t)
            except Exception as e:
                print(f"(تعذّر جلب البيانات الأساسية/الأخبار: {e})")

        print_decision_summary(ticker, score, verdict, extra, fund, args.amount)

        if args.backtest:
            trades = backtest_strategy(df, entry_threshold=args.entry_threshold, exit_threshold=args.exit_threshold)
            stats = summarize_backtest(trades, df)
            print_backtest_results(ticker, trades, stats)

        print("\n--- تفاصيل المؤشرات ---")
        for n in notes:
            print(f"  - {n}")

        print(f"\nنسبة المخاطرة إلى العائد: {rr_note}")

        if fund:
            print_fundamentals(fund)

        if args.csv:
            csv_path = f"/mnt/user-data/outputs/{ticker.replace('.', '_')}_data.csv"
            df.to_csv(csv_path)
            print(f"\n💾 تم حفظ البيانات: {csv_path}")

        if args.plot:
            chart_path = plot_ticker(ticker, df)
            print(f"🖼️  تم حفظ الرسم البياني: {chart_path}")

        print()

    print("=" * 65)
    print("ملاحظات مهمة:")
    print("  • سوقا دبي وأبوظبي يتداولان من الاثنين للجمعة، 10:00 ص - 3:00 م بتوقيت الإمارات (GMT+4).")
    print("    شغّل السكربت بعد إغلاق الجلسة (~3:00 م) للحصول على إغلاق اليوم نفسه بشكل مؤكد.")
    print("  • إجماع المحللين/الأخبار غير متوفرة دائماً لأسهم DFM/ADX عبر Yahoo Finance.")
    print("  • Fibonacci Bollinger Bands و Nadaraya-Watson هنا تقريبات شائعة، قد تختلف قليلاً عن TradingView.")
    print("  • هذا التحليل لأغراض تعليمية فقط وليس توصية استثمارية.")


if __name__ == "__main__":
    main()
