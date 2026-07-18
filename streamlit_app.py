# -*- coding: utf-8 -*-
"""
تطبيق ويب لتحليل أسهم الإمارات - مبني على Streamlit
يعيد استخدام كل منطق التحليل من uae_stocks_analysis.py (لازم يكون بنفس المجلد).
"""

import streamlit as st
import pandas as pd
import numpy as np

from uae_stocks_analysis import (
    DEFAULT_TICKERS, analyze_ticker, build_composite_signal,
    fetch_fundamental_info, fetch_latest_quote, decision_from_score,
    backtest_strategy, summarize_backtest, yf,
)

st.set_page_config(page_title="تحليل أسهم الإمارات", layout="centered")
st.title("📊 تحليل أسهم الإمارات (DFM / ADX)")
st.caption("مبني على بيانات Yahoo Finance - لأغراض تعليمية فقط وليس توصية استثمارية")

with st.sidebar:
    st.header("الإعدادات")
    tickers_input = st.text_area(
        "رموز الأسهم (رمز بكل سطر)",
        value="\n".join(DEFAULT_TICKERS),
        height=160,
    )
    tickers = [t.strip() for t in tickers_input.splitlines() if t.strip()]

    days = st.slider("عدد أيام البيانات التاريخية", 200, 800, 400, step=50)
    amount = st.number_input("المبلغ المخصص لكل سهم (AED)", min_value=100, value=10000, step=500)
    run_backtest = st.checkbox("تشغيل محاكاة تاريخية (Backtest) أيضاً", value=False)
    fetch_fund = st.checkbox("جلب إجماع المحللين والأخبار (أبطأ)", value=True)

    run_button = st.button("🔍 شغّل التحليل", type="primary", use_container_width=True)

if run_button:
    if not tickers:
        st.warning("ضيف رمز سهم واحد على الأقل.")
        st.stop()

    for ticker in tickers:
        st.markdown(f"## {ticker}")
        with st.spinner(f"جاري جلب وتحليل بيانات {ticker}..."):
            df = analyze_ticker(ticker, days)

        if df is None:
            st.error(f"لا توجد بيانات لهذا الرمز. تأكد من صيغته (مثل EMAAR.AE أو ADIB.AB).")
            continue

        last = df.iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("آخر إغلاق", f"{last['Close']:.2f} AED", f"{change_pct:+.2f}%")

        try:
            quote = fetch_latest_quote(yf.Ticker(ticker))
        except Exception:
            quote = None
        if quote:
            col2.metric("أحدث سعر متاح", f"{quote['price']:.2f} AED")
            st.caption(f"بتاريخ ووقت: {quote['timestamp']}  |  {quote['source']}")

        score, verdict, notes, rr_note, extra = build_composite_signal(df, ticker)
        col3.metric("درجة الإشارة", f"{score:+.2f}", verdict)

        st.info(f"**القرار المقترح:** {decision_from_score(score)}")

        fund = {}
        if fetch_fund:
            try:
                fund = fetch_fundamental_info(yf.Ticker(ticker))
            except Exception:
                fund = {}

        price = extra["price"]
        supports, resistances = extra["supports"], extra["resistances"]
        if fund.get("target_mean"):
            target = fund["target_mean"]
            target_source = f"إجماع {fund.get('num_analysts') or '؟'} محلل"
        elif resistances:
            target = resistances[-1]
            target_source = "تقدير فني تقريبي (أبعد مقاومة)"
        else:
            target = price * 1.10
            target_source = "تقدير عام تقريبي"
        upside = (target - price) / price * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("السعر المستهدف (12 شهر)", f"{target:.2f} AED", f"{upside:+.1f}%")
        c1.caption(target_source)
        shares = int(amount // price) if price > 0 else 0
        c2.metric("عدد الأسهم الممكن شراؤها", f"{shares:,}", f"بمبلغ {amount:,.0f} AED")
        c3.metric("نسبة المخاطرة/العائد", rr_note.split("(")[0].strip())

        with st.expander("📋 تفاصيل المؤشرات الفنية"):
            for n in notes:
                st.write(f"- {n}")

        st.line_chart(df[["Close", "EMA20", "EMA50", "EMA200"]].dropna())

        if fund.get("news"):
            with st.expander("📰 أهم الأخبار"):
                for item in fund["news"][:3]:
                    title = item.get("title") or item.get("content", {}).get("title")
                    link = item.get("link") or item.get("content", {}).get("clickThroughUrl", {}).get("url")
                    if title:
                        st.write(f"- [{title}]({link})" if link else f"- {title}")

        if run_backtest:
            with st.spinner("جاري تشغيل المحاكاة التاريخية..."):
                trades = backtest_strategy(df)
                stats = summarize_backtest(trades, df)
            with st.expander("🔁 نتائج المحاكاة التاريخية (Backtest)"):
                if not trades or stats is None:
                    st.write("لا توجد صفقات كافية لتقييم الاستراتيجية بهذه الفترة.")
                else:
                    b1, b2, b3 = st.columns(3)
                    b1.metric("عدد الصفقات", stats["num_trades"])
                    b2.metric("نسبة الصفقات الرابحة", f"{stats['win_rate']:.1f}%")
                    b3.metric("العائد الإجمالي", f"{stats['total_return_pct']:+.1f}%")
                    st.write(f"عائد الشراء والاحتفاظ لنفس الفترة: {stats['buy_hold_return_pct']:+.1f}%")
                    st.dataframe(pd.DataFrame(trades[-10:]))

        st.divider()

    st.caption("⚠️ هذا التحليل لأغراض تعليمية فقط وليس توصية استثمارية. تحقق دائماً من مصادر رسمية قبل اتخاذ قرارات مالية.")
else:
    st.write("اضبط الإعدادات من القائمة الجانبية واضغط **شغّل التحليل** للبدء.")
