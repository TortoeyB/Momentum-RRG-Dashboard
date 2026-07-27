"""
diagnose.py — ตรวจสอบว่า sub-score แต่ละตัวมาจากไหน ด้วยข้อมูลจริง

วางไว้ใน scripts/ แล้วรัน:
    python scripts/diagnose.py DAPP
    python scripts/diagnose.py DAPP COPX NFLX

จะพิมพ์
  1) ค่า indicator ดิบ 10 แท่งสุดท้าย
  2) การถอด sub-score ทีละชิ้น (ตอบว่า Hull/Cipher ได้คะแนนนี้เพราะอะไร)
  3) สถิติว่า threshold WT/divergence ยิงบ่อยแค่ไหนในประวัติทั้งหมด
"""

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import (WT_OB, WT_OS, WT_GOLD, DIV_OB, DIV_OS,
                        compute_all, slope_up)
from scoring import WEIGHTS, bar_shock_series, price_structure, score_series

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

PERIOD = os.environ.get("DIAGNOSE_PERIOD", "3y")


def fetch(sym: str, period: str = None) -> pd.DataFrame:
    df = yf.download(sym, period=period or PERIOD, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def explain(sym: str) -> None:
    df = fetch(sym)
    if len(df) < 220:
        print(f"[{sym}] ข้อมูลไม่พอ ({len(df)} แท่ง)")
        return

    ind = compute_all(df)
    sc = score_series(df)
    last_i = -1
    d = df.iloc[last_i]
    i = ind.iloc[last_i]
    s = sc.iloc[last_i]

    print("=" * 78)
    print(f"{sym}   ข้อมูล ณ {df.index[last_i].date()}   close={float(d['Close']):.4f}")
    print("=" * 78)

    # ---------- 1. ค่าดิบ ----------
    cols = ["close", "hma9", "hma20", "ma50", "ma200", "wt1", "wt2", "mf", "rsi", "adx"]
    print("\n[1] indicator 10 แท่งสุดท้าย")
    print(ind[cols].tail(10).round(3).to_string())

    # ---------- 2. ถอด sub-score ----------
    print("\n[2] ถอด sub-score")

    hma9, hma20, close = float(i["hma9"]), float(i["hma20"]), float(i["close"])
    a = 1.0 if hma9 > hma20 else -1.0
    b = 0.5 if close > hma9 else -0.5
    slope9 = bool(slope_up(ind["hma9"], 2).iloc[last_i])
    cc = 0.5 if slope9 else -0.5
    print(f"  Hull = {float(s['hull_sc']):+.1f}")
    print(f"    hma9({hma9:.4f}) {'>' if a > 0 else '<'} hma20({hma20:.4f})   -> {a:+.1f}")
    print(f"    close({close:.4f}) {'>' if b > 0 else '<'} hma9            -> {b:+.1f}")
    print(f"    hma9 slope(2) {'ขึ้น' if slope9 else 'ลง'}                  -> {cc:+.1f}")
    print("    ^ ribbon แดงบน TradingView = HMA slope ไม่ใช่ hma9<hma20 — คนละตัววัด")

    wt1, wt2 = float(i["wt1"]), float(i["wt2"])
    buy3 = bool(ind["wt_buy"].iloc[-3:].any())
    sell3 = bool(ind["wt_sell"].iloc[-3:].any())
    bdiv5 = bool(ind["bull_div"].iloc[-5:].any())
    sdiv5 = bool(ind["bear_div"].iloc[-5:].any())
    gold5 = bool(ind["gold_buy"].iloc[-5:].any())
    print(f"\n  Cipher B = {float(s['cipher_sc']):+.1f}")
    print(f"    wt1={wt1:+.2f}  wt2={wt2:+.2f}  spread={wt1 - wt2:+.2f}")
    print(f"    wt_buy(3d)={buy3}  [ต้อง cross up และ wt2<={WT_OS}]")
    print(f"    wt_sell(3d)={sell3} [ต้อง cross dn และ wt2>=+{WT_OB}]")
    print(f"    bull_div(5d)={bdiv5}  bear_div(5d)={sdiv5}  gold(5d)={gold5}")
    print(f"    money flow={float(i['mf']):+.2f} -> {'+0.5' if i['mf'] > 0 else '-0.5'}")

    print(f"\n  MA50  = {float(s['ma50_sc']):+.2f}   (ห่างเส้น {100 * (close / float(i['ma50']) - 1):+.2f}%)")
    print(f"  MA200 = {float(s['ma200_sc']):+.2f}   (ห่างเส้น {100 * (close / float(i['ma200']) - 1):+.2f}%)")

    raw = sum(WEIGHTS[k] * float(s[f"{k}_sc"]) for k in WEIGHTS)
    print(f"\n  raw = {raw:+.4f}  ->  base = {50 + raw * 25:.2f}"
          f"  ->  x{float(s['adx_mult']):.2f} (ADX {float(i['adx']):.1f})"
          f"  ->  score = {float(s['score']):.1f}")

    # ---------- 3. แท่งล่าสุด + โครงสร้าง ----------
    shock = bar_shock_series(df).iloc[last_i]
    st = price_structure(df)
    print(f"\n[3] แท่งล่าสุด: ret={float(shock['bar_ret']):+.2%}  "
          f"ปิดที่ {float(shock['bar_loc']):.0%} ของ range  "
          f"hard_down={bool(shock['hard_down'])}")
    print(f"    โครงสร้าง 10 วัน: {st['label']}  (code={st['code']})")

    # ---------- 4. threshold ยิงบ่อยแค่ไหน ----------
    n = len(ind)
    q = ind["wt2"].dropna().quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    print(f"\n[4] สถิติ {n} แท่ง — WT2 percentile")
    print("    " + "  ".join(f"p{int(k * 100)}={v:+.1f}" for k, v in q.items()))
    for name, cond in [
        (f"wt2 <= {WT_OS} (โซนจุดเขียว)", ind["wt2"] <= WT_OS),
        (f"wt2 >= +{WT_OB} (โซนจุดแดง)", ind["wt2"] >= WT_OB),
        (f"wt2 <= {WT_GOLD} (โซน gold)", ind["wt2"] <= WT_GOLD),
        (f"wt2 <= {DIV_OS} (โซน bull div)", ind["wt2"] <= DIV_OS),
        (f"wt2 >= +{DIV_OB} (โซน bear div)", ind["wt2"] >= DIV_OB),
    ]:
        k = int(cond.sum())
        print(f"    {name:32s} {k:5d} วัน ({100 * k / n:5.1f}%)")
    for name, col in [("wt_buy (จุดเขียว)", "wt_buy"), ("wt_sell (จุดแดง)", "wt_sell"),
                      ("bull_div", "bull_div"), ("bear_div", "bear_div"),
                      ("gold_buy", "gold_buy")]:
        k = int(ind[col].sum())
        print(f"    {name:32s} {k:5d} ครั้ง")

    # spread scale — ใช้ตั้งค่า WT_SPREAD_NORM ใน scoring.py
    sp = (ind["wt1"] - ind["wt2"]).abs().dropna()
    print(f"\n    |wt1-wt2|: median={sp.median():.2f}  p90={sp.quantile(0.9):.2f}"
          f"  -> แนะนำ WT_SPREAD_NORM ~= {sp.quantile(0.9):.0f}")
    print()


if __name__ == "__main__":
    syms = sys.argv[1:] or ["DAPP"]
    for sym in syms:
        try:
            explain(sym)
        except Exception as e:  # noqa: BLE001
            print(f"[{sym}] ERROR: {type(e).__name__}: {e}")
