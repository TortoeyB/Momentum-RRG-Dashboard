"""
scoring.py — แปลง indicator เป็นคะแนน 0–100 + buy/sell signal

โครงสร้างคะแนน (ตามที่ตกลงกัน):
  sub-score แต่ละตัวอยู่ในช่วง -2..+2
    MA200  (20%): เหนือ/ใต้เส้น ±1, slope ±1
    MA50   (20%): เหนือ/ใต้ ±1, slope ±1, ยืดเกิน 12% โดนหักกลับ 1
    Hull   (25%): HMA9 vs HMA20 ±1, slope HMA9 ±1
    Cipher (35%): WT cross ±1.5 (จุดเขียว/แดง ภายใน 3 วัน), divergence ±1,
                  money flow ±0.5, gold buy = เต็ม +2
  raw = ผลรวมถ่วงน้ำหนัก (-2..+2) → base = 50 + raw*25
  ADX คูณส่วนเบี่ยงจาก 50:  >=35 → ×1.25, 20–35 → ×1.0, <20 → ×0.6

Buy checklist (0–10):
  +2 quadrant Improving หรือเพิ่งข้ามเข้า Leading
  +2 โครงสร้างราคา Higher low / Higher high
  +1 bullish pattern ตรงจุดสำคัญ (ใกล้ MA50/MA200 หรือหลัง divergence)
  +2 WT ตัดขึ้นในโซน oversold (ภายใน 3 วัน)
  +2 bullish divergence บน WT (ภายใน 5 วัน)
  +1 money flow เขียว
  (gold buy → หมวด Cipher B ได้เต็ม 5 ทันที)
  >=7 = BUY, 4–6 = WATCH   |   ฝั่ง SELL ใช้เงื่อนไขกลับด้าน
"""

import numpy as np
import pandas as pd

from indicators import compute_all, slope_up

# ----------------------------------------------------------------
# คะแนนรายวัน (ทั้ง series เพื่อใช้ลากหาง RRG)
# ----------------------------------------------------------------

WEIGHTS = {"cipher": 0.35, "hull": 0.25, "ma50": 0.20, "ma200": 0.20}

def score_series(df: pd.DataFrame) -> pd.DataFrame:
    """คืน DataFrame: score, sub-scores, adx_mult ต่อวัน"""
    ind = compute_all(df)
    c = ind["close"]

    ma200_sc = np.where(c > ind["ma200"], 1, -1) + np.where(slope_up(ind["ma200"], 5), 1, -1)

    ext = (c / ind["ma50"] - 1).abs() > 0.12
    ma50_sc = (np.where(c > ind["ma50"], 1, -1)
               + np.where(slope_up(ind["ma50"], 3), 1, -1)
               - np.where(ext & (c > ind["ma50"]), 1, 0)
               + np.where(ext & (c < ind["ma50"]), 1, 0))
    ma50_sc = np.clip(ma50_sc, -2, 2)

    hull_sc = (np.where(ind["hma9"] > ind["hma20"], 1, -1)
               + np.where(slope_up(ind["hma9"], 2), 1, -1))

    buy3 = ind["wt_buy"].rolling(3, min_periods=1).max().astype(bool)
    sell3 = ind["wt_sell"].rolling(3, min_periods=1).max().astype(bool)
    bdiv5 = ind["bull_div"].rolling(5, min_periods=1).max().astype(bool)
    sdiv5 = ind["bear_div"].rolling(5, min_periods=1).max().astype(bool)
    gold5 = ind["gold_buy"].rolling(5, min_periods=1).max().astype(bool)

    cipher_sc = (np.where(buy3, 1.5, np.where(sell3, -1.5,
                 np.where(ind["wt1"] > ind["wt2"], 0.5, -0.5)))
                 + np.where(bdiv5, 1, 0) - np.where(sdiv5, 1, 0)
                 + np.where(ind["mf"] > 0, 0.5, -0.5))
    cipher_sc = np.where(gold5, 2.0, np.clip(cipher_sc, -2, 2))

    raw = (WEIGHTS["cipher"] * cipher_sc + WEIGHTS["hull"] * hull_sc
           + WEIGHTS["ma50"] * ma50_sc + WEIGHTS["ma200"] * ma200_sc)
    base = 50 + raw * 25

    mult = np.where(ind["adx"] >= 35, 1.25, np.where(ind["adx"] < 20, 0.6, 1.0))
    score = np.clip(50 + (base - 50) * mult, 0, 100)

    out = pd.DataFrame(index=df.index)
    out["score"] = score
    out["cipher_sc"] = np.round(cipher_sc, 1)
    out["hull_sc"] = hull_sc
    out["ma50_sc"] = ma50_sc
    out["ma200_sc"] = ma200_sc
    out["adx_mult"] = mult
    for col in ["wt_buy", "wt_sell", "bull_div", "bear_div", "gold_buy", "mf",
                "ma50", "ma200", "hma9", "hma20", "doji", "hammer", "shooting_star",
                "bull_engulf", "bear_engulf", "morning_star", "evening_star", "adx",
                "di_plus", "di_minus"]:
        out[col] = ind[col]
    return out

# ----------------------------------------------------------------
# quadrant
# ----------------------------------------------------------------

def quadrant(score: float, delta: float) -> str:
    if score >= 50:
        return "Leading" if delta >= 0 else "Weakening"
    return "Improving" if delta >= 0 else "Lagging"

# ----------------------------------------------------------------
# โครงสร้างราคา 10 วัน
# ----------------------------------------------------------------

def price_structure(df: pd.DataFrame) -> dict:
    """คืน {'label','tone','code'} — tone: 1 เขียว, 0 แดง, 2 น้ำเงิน(รอยืนยัน)"""
    lows = df["Low"].tail(10).to_numpy()
    highs = df["High"].tail(10).to_numpy()
    close = float(df["Close"].iloc[-1])
    imin = int(np.argmin(lows))
    imax = int(np.argmax(highs))

    if imax >= 8 and lows[5:].min() > lows[:5].min():
        return {"label": "Higher high", "tone": 1, "code": "higher_high"}
    if imin >= 8:
        return {"label": "New 10D low", "tone": 0, "code": "new_low"}
    if lows[5:].min() > lows[:5].min() * 1.004:
        return {"label": "Higher low", "tone": 1, "code": "higher_low"}
    if close >= lows[imin] * 1.03:
        return {"label": "เด้งจาก low · รอยืนยัน", "tone": 2, "code": "bounce"}
    return {"label": "Lower low", "tone": 0, "code": "lower_low"}

# ----------------------------------------------------------------
# pattern ที่ "มีนัย" — ใกล้ MA50/MA200 หรือหลัง divergence
# ----------------------------------------------------------------

PATTERN_TH = {
    "hammer": ("Hammer", 1), "bull_engulf": ("Bull engulfing", 1),
    "morning_star": ("Morning star", 1), "shooting_star": ("Shooting star", 0),
    "bear_engulf": ("Bear engulfing", 0), "evening_star": ("Evening star", 0),
    "doji": ("Doji (ลังเล)", 2),
}

def significant_pattern(sc: pd.DataFrame, lookback: int = 3) -> dict | None:
    tail = sc.tail(lookback)
    close = float(sc["close"].iloc[-1]) if "close" in sc else None
    near_ma = False
    if close and not np.isnan(sc["ma50"].iloc[-1]):
        for ma in ("ma50", "ma200"):
            v = sc[ma].iloc[-1]
            if not np.isnan(v) and abs(close / v - 1) <= 0.025:
                near_ma = True
    recent_div = bool(sc["bull_div"].tail(5).any() or sc["bear_div"].tail(5).any())

    for key in ["bull_engulf", "morning_star", "hammer",
                "bear_engulf", "evening_star", "shooting_star", "doji"]:
        if bool(tail[key].any()):
            label, tone = PATTERN_TH[key]
            if key == "doji":
                return {"label": label, "tone": tone, "code": key}
            if near_ma:
                label += " @MA"
            elif recent_div and tone == 1:
                label += " + divergence"
            return {"label": label, "tone": tone, "code": key,
                    "significant": near_ma or recent_div}
    return None

# ----------------------------------------------------------------
# Buy / Sell checklist
# ----------------------------------------------------------------

def signal(sc: pd.DataFrame, quad_now: str, quad_prev: str,
           structure: dict, pattern: dict | None) -> dict | None:
    last = sc.iloc[-1]
    mf_green = bool(last["mf"] > 0)
    wt_buy3 = bool(sc["wt_buy"].tail(3).any())
    wt_sell3 = bool(sc["wt_sell"].tail(3).any())
    bdiv5 = bool(sc["bull_div"].tail(5).any())
    sdiv5 = bool(sc["bear_div"].tail(5).any())
    gold5 = bool(sc["gold_buy"].tail(5).any())

    # ---------- BUY ----------
    b = []
    fresh_leading = quad_now == "Leading" and quad_prev == "Improving"
    b.append((f"Quadrant: {quad_now}" + (" (เพิ่งข้ามจาก Improving)" if fresh_leading else ""),
              quad_now == "Improving" or fresh_leading, 2))
    b.append((f"โครงสร้าง: {structure['label']}",
              structure["code"] in ("higher_low", "higher_high"), 2))
    b.append((("Pattern: " + pattern["label"]) if (pattern and pattern["tone"] == 1)
              else "ไม่มี bullish pattern", bool(pattern and pattern["tone"] == 1), 1))
    if gold5:
        b.append(("GOLD BUY (WT2≤-75 + RSI<30 + divergence)", True, 5))
        b.append(("WT ตัดขึ้นในโซน oversold", wt_buy3, 0))
        b.append(("Money flow เขียว", mf_green, 0))
    else:
        b.append(("WT ตัดขึ้นในโซน oversold", wt_buy3, 2))
        b.append(("Bullish divergence บน WT", bdiv5, 2))
        b.append(("Money flow เขียว", mf_green, 1))
    buy_score = sum(p for _, ok, p in b if ok)

    # ---------- SELL ----------
    s = []
    fresh_lagging = quad_now == "Lagging" and quad_prev == "Weakening"
    s.append((f"Quadrant: {quad_now}" + (" (เพิ่งหลุดเป็น Lagging)" if fresh_lagging else ""),
              quad_now == "Weakening" or fresh_lagging, 2))
    s.append((f"โครงสร้าง: {structure['label']}",
              structure["code"] in ("lower_low", "new_low"), 2))
    s.append((("Pattern: " + pattern["label"]) if (pattern and pattern["tone"] == 0)
              else "ไม่มี bearish pattern", bool(pattern and pattern["tone"] == 0), 1))
    s.append(("WT ตัดลงเหนือ overbought", wt_sell3, 2))
    s.append(("Bearish divergence บน WT", sdiv5, 2))
    s.append(("Money flow แดง", not mf_green, 1))
    sell_score = sum(p for _, ok, p in s if ok)

    def pack(grade, score, items):
        return {"grade": grade, "score": int(score),
                "checklist": [[t, bool(ok)] for t, ok, _ in items]}

    if buy_score >= 7:
        return pack("BUY", min(buy_score, 10), b)
    if sell_score >= 7:
        return pack("SELL", min(sell_score, 10), s)
    if buy_score >= 4 and buy_score >= sell_score:
        return pack("WATCH", buy_score, b)
    if sell_score >= 4:
        return pack("REDUCE", sell_score, s)
    if quad_now == "Leading" and float(last["score"]) >= 60:
        return {"grade": "HOLD", "score": 0, "checklist": []}
    return None
