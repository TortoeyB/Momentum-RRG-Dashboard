"""
scoring.py — แปลง indicator เป็นคะแนน 0–100 + buy/sell signal

โครงสร้างคะแนน:
  sub-score แต่ละตัวอยู่ในช่วง -2..+2
    MA200  (20%): เหนือ/ใต้เส้น ±1 (graded ±10%), slope ±1
    MA50   (20%): เหนือ/ใต้ ±1 (graded ±5%), slope ±1, ยืดเกิน 12% โดนหักกลับ 1
    Hull   (25%): HMA9 vs HMA20 ±1, ราคาเทียบ HMA9 ±0.5, slope HMA9 ±0.5
    Cipher (35%): WT cross ±1.5 (ภายใน 3 วัน) | ถ้าไม่มี cross ใช้ spread แบบ graded ±0.5,
                  divergence ±1, money flow ±0.5, gold buy = เต็ม +2
  raw = ผลรวมถ่วงน้ำหนัก (-2..+2) → base = 50 + raw*25
  ADX คูณส่วนเบี่ยงจาก 50:  >=35 → ×1.25, 20–35 → ×1.0, <20 → ×0.6

Buy checklist (0–10):
  +2 quadrant Improving หรือเพิ่งข้ามเข้า Leading
  +2 โครงสร้างราคา Higher low / Higher high
  +1 bullish pattern ตรงจุดสำคัญ (ใกล้ MA50/MA200 หรือหลัง divergence)
  +2 WT ตัดขึ้นในโซน oversold (ภายใน 3 วัน)
  +2 bullish divergence บน WT (ภายใน 5 วัน)
  +1 money flow เขียว
  (gold buy → หมวด Cipher B ได้เต็ม 5 ทันที ถ้าแท่งล่าสุดไม่ใช่ hard down)
  >=7 = BUY
  5–6 + โครงสร้างยืนยัน (Higher low/high) + money flow ไม่แดง = ACCUM (ทยอยซื้อ)
  4–6 อื่นๆ = WATCH   |   ฝั่ง SELL ใช้เงื่อนไขกลับด้าน (SELL / REDUCE)

  BAR-LEVEL VETO: ถ้าแท่งล่าสุด "hard down" (ปิด <= -3% และปิดในโซนล่าง 25%
  ของ range วัน) → บล็อก BUY/ACCUM ทั้งหมด ดาวน์เกรดลงมาเป็น WATCH
"""

import numpy as np
import pandas as pd

from indicators import compute_all, slope_up

# ----------------------------------------------------------------
# ค่าคงที่ปรับจูน (แก้ที่เดียว มีผลทั้งระบบ)
# ----------------------------------------------------------------

WEIGHTS = {"cipher": 0.35, "hull": 0.25, "ma50": 0.20, "ma200": 0.20}

HARD_DOWN_RET = -0.03   # ผลตอบแทนแท่งเดียวที่ถือว่ารุนแรง
HARD_DOWN_LOC = 0.25    # ปิดในโซนล่าง x% ของ range วัน (0 = ปิดที่ low พอดี)
WT_SPREAD_NORM = 5.0    # ตัวหาร normalize spread WT1-WT2 ให้เป็นคะแนน graded

# ตัวคูณ ADX — เดิมเป็นบันได 0.6 / 1.0 / 1.25 ตัดที่ 20 กับ 35
# ทำให้ ADX ต่างกัน 1 หน่วยรอบเส้นแบ่ง ทำคะแนนต่างกันได้ถึง ~17 (เคส SMH vs SOXX)
# เปลี่ยนเป็นไล่ระดับต่อเนื่องผ่าน np.interp — เจตนาเดิมคงอยู่ แต่ไม่มีหน้าผา
ADX_KNOTS_X = [10.0, 20.0, 35.0, 45.0]
ADX_KNOTS_Y = [0.60, 0.80, 1.10, 1.25]

# ----------------------------------------------------------------
# bar-level shock: มองแท่งล่าสุดตรง ๆ
# ----------------------------------------------------------------

def bar_shock_series(df: pd.DataFrame) -> pd.DataFrame:
    """คืน DataFrame: bar_ret, bar_loc, hard_down ต่อวัน

    bar_loc = ตำแหน่งราคาปิดใน range ของวัน (0 = ปิดที่ low, 1 = ปิดที่ high)
    hard_down = แท่งร่วงแรงและปิดใกล้ low → โครงสร้างระยะสั้นเพิ่งพัง
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    rng = (high - low).replace(0, np.nan)
    bar_ret = close.pct_change()
    bar_loc = ((close - low) / rng).fillna(0.5).clip(0, 1)

    out = pd.DataFrame(index=df.index)
    out["bar_ret"] = bar_ret.fillna(0.0)
    out["bar_loc"] = bar_loc
    out["hard_down"] = (out["bar_ret"] <= HARD_DOWN_RET) & (out["bar_loc"] <= HARD_DOWN_LOC)
    return out

# ----------------------------------------------------------------
# คะแนนรายวัน (ทั้ง series เพื่อใช้ลากหาง RRG)
# ----------------------------------------------------------------

def score_series(df: pd.DataFrame) -> pd.DataFrame:
    """คืน DataFrame: score, sub-scores, adx_mult ต่อวัน"""
    ind = compute_all(df)
    c = ind["close"]

    # MA200: ตำแหน่ง = % ห่างจากเส้น (±10% = เต็ม ±1) + slope ±1
    pct200 = (c / ind["ma200"] - 1)
    ma200_sc = np.clip(np.clip(pct200 / 0.10, -1, 1)
                       + np.where(slope_up(ind["ma200"], 5), 1, -1), -2, 2)

    # MA50: ตำแหน่ง = % ห่าง (±5% = เต็ม ±1) + slope ±1
    #       ยืดเกิน 12% โดนหักไล่ระดับ (เตือน mean reversion)
    pct50 = (c / ind["ma50"] - 1)
    over = np.clip((pct50 - 0.12) / 0.06, 0, 1)     # ยืดขึ้นเกิน
    under = np.clip((-pct50 - 0.12) / 0.06, 0, 1)   # ทิ้งลงเกิน
    ma50_sc = np.clip(np.clip(pct50 / 0.05, -1, 1)
                      + np.where(slope_up(ind["ma50"], 3), 1, -1)
                      - over + under, -2, 2)

    # Hull: HMA9 vs HMA20 (±1) + ราคาเทียบ HMA9 (±0.5) + slope (±0.5)
    hull_sc = np.clip(np.where(ind["hma9"] > ind["hma20"], 1.0, -1.0)
                      + np.where(c > ind["hma9"], 0.5, -0.5)
                      + np.where(slope_up(ind["hma9"], 2), 0.5, -0.5), -2, 2)

    buy3 = ind["wt_buy"].rolling(3, min_periods=1).max().astype(bool)
    sell3 = ind["wt_sell"].rolling(3, min_periods=1).max().astype(bool)
    bdiv5 = ind["bull_div"].rolling(5, min_periods=1).max().astype(bool)
    sdiv5 = ind["bear_div"].rolling(5, min_periods=1).max().astype(bool)
    gold5 = ind["gold_buy"].rolling(5, min_periods=1).max().astype(bool)

    shock = bar_shock_series(df)
    hard_down = shock["hard_down"].to_numpy()

    # FIX: ตอนไม่มี WT cross เดิมใช้ binary ±0.5 (ไม่สนว่า WT หักหัวลงชันแค่ไหน)
    #      เปลี่ยนเป็น graded ตาม spread → spread ที่กำลังหุบจะได้คะแนนน้อยลงจริง
    wt_spread = (ind["wt1"] - ind["wt2"]).to_numpy(dtype=float)
    neutral_sc = np.clip(wt_spread / WT_SPREAD_NORM, -0.5, 0.5)

    cipher_sc = (np.where(buy3, 1.5, np.where(sell3, -1.5, neutral_sc))
                 + np.where(bdiv5, 1, 0) - np.where(sdiv5, 1, 0)
                 + np.where(ind["mf"] > 0, 0.5, -0.5))

    # FIX: gold buy เดิม override เป็น +2 แบบไม่มีเงื่อนไข ทำให้ถ้าเกิด gold buy
    #      แล้ววันถัดมาร่วงแรง ระบบยังให้ +2 อยู่ → ตอนนี้ยกเลิก override ถ้าแท่ง
    #      ล่าสุดเป็น hard down
    gold_ok = gold5.to_numpy() & ~hard_down
    cipher_sc = np.where(gold_ok, 2.0, np.clip(cipher_sc, -2, 2))

    # NaN-safe: ตัด sub ที่เป็น NaN (เช่น MA200 ของหุ้น IPO ใหม่ที่ประวัติ < 200 วัน)
    # ออกจากสมการ แล้ว renormalize น้ำหนักที่เหลือให้รวมเป็น 1
    subs = {"cipher": cipher_sc, "hull": hull_sc, "ma50": ma50_sc, "ma200": ma200_sc}
    num = np.zeros(len(c))
    den = np.zeros(len(c))
    for k, arr in subs.items():
        arr = np.asarray(arr, dtype=float)
        ok = ~np.isnan(arr)
        num += WEIGHTS[k] * np.where(ok, arr, 0.0)
        den += WEIGHTS[k] * ok
    raw = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    base = 50 + raw * 25

    adx_v = np.asarray(ind["adx"], dtype=float)
    mult = np.interp(adx_v, ADX_KNOTS_X, ADX_KNOTS_Y)
    # ADX ยังเป็น NaN ช่วง warm-up -> ใช้ 1.0 เท่าพฤติกรรมเดิม
    mult = np.where(np.isnan(adx_v), 1.0,
                    np.clip(mult, ADX_KNOTS_Y[0], ADX_KNOTS_Y[-1]))
    score = np.clip(50 + (base - 50) * mult, 0, 100)

    out = pd.DataFrame(index=df.index)
    out["score"] = score
    out["cipher_sc"] = np.round(cipher_sc, 1)
    out["hull_sc"] = hull_sc
    out["ma50_sc"] = ma50_sc
    out["ma200_sc"] = ma200_sc
    out["adx_mult"] = mult
    # ส่ง bar-level context เข้าไปใน sc เพื่อให้ signal() อ่านได้โดยไม่ต้องรับ df
    out["bar_ret"] = shock["bar_ret"]
    out["bar_loc"] = shock["bar_loc"]
    out["hard_down"] = shock["hard_down"]
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

    recent = lows[5:]
    prev_min = lows[:5].min()
    # FIX: เดิมเช็คแค่ min(5 แท่งหลัง) > min(5 แท่งแรก) ทำให้ถ้า 5 แท่งแรกมี
    #      capitulation low ลึก ๆ แท่งไหนก็ตามที่สูงกว่าจะถูกนับเป็น Higher low
    #      แม้แท่งล่าสุดจะเป็น low ต่ำสุดของ 5 แท่งหลังเองก็ตาม
    #      เพิ่มเงื่อนไข rising: low ต้องกำลังไต่ขึ้นจริง ไม่ใช่แค่สูงกว่าก้นเก่า
    #      (แท่งล่าสุดต้องไม่ใช่ก้นของครึ่งหลังเอง)
    rising = int(np.argmin(recent)) != len(recent) - 1

    if imax >= 8 and recent.min() > prev_min and rising:
        return {"label": "Higher high", "tone": 1, "code": "higher_high"}
    if imin >= 8:
        return {"label": "New 10D low", "tone": 0, "code": "new_low"}
    if recent.min() > prev_min * 1.004 and rising:
        return {"label": "Higher low", "tone": 1, "code": "higher_low"}
    if recent.min() > prev_min * 1.004 and not rising:
        return {"label": "Low ยกตัว แต่แท่งล่าสุดทำ low ใหม่ · รอยืนยัน",
                "tone": 2, "code": "bounce"}
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

def eval_at(sc: pd.DataFrame, df: pd.DataFrame, ss: pd.Series):
    """ประเมิน quadrant/structure/pattern/signal ณ แท่งสุดท้ายของข้อมูลที่ให้มา"""
    d5 = float(ss.iloc[-1] - ss.iloc[-6])
    d5p = float(ss.iloc[-6] - ss.iloc[-11])
    q_now = quadrant(float(ss.iloc[-1]), d5)
    q_prev = quadrant(float(ss.iloc[-6]), d5p)
    struct = price_structure(df)
    sc_t = sc.iloc[-60:].copy()
    sc_t["close"] = df["Close"].iloc[-60:]
    patt = significant_pattern(sc_t)
    sig = signal(sc, q_now, q_prev, struct, patt)
    return sig, struct, patt, q_now, q_prev


def signal_with_age(sc: pd.DataFrame, df: pd.DataFrame, ss: pd.Series,
                    max_back: int = 12):
    """เหมือน eval_at แต่เพิ่ม 'สัญญาณเกรดนี้เริ่มตั้งแต่วันไหน' (since/days)"""
    sig, struct, patt, q_now, q_prev = eval_at(sc, df, ss)
    if sig:
        days, date = 0, df.index[-1]
        for k in range(1, max_back + 1):
            n = min(len(df), len(sc)) - k
            m = len(ss) - k
            if n < 220 or m < 12:
                break
            s_k, *_ = eval_at(sc.iloc[:n], df.iloc[:n], ss.iloc[:m])
            if s_k and s_k["grade"] == sig["grade"]:
                days, date = k, df.index[n - 1]
            else:
                break
        sig["since"] = date.strftime("%d %b")
        sig["days"] = days
    return sig, struct, patt, q_now, q_prev


def signal(sc: pd.DataFrame, quad_now: str, quad_prev: str,
           structure: dict, pattern: dict | None) -> dict | None:
    last = sc.iloc[-1]
    mf_green = bool(last["mf"] > 0)
    wt_buy3 = bool(sc["wt_buy"].tail(3).any())
    wt_sell3 = bool(sc["wt_sell"].tail(3).any())
    bdiv5 = bool(sc["bull_div"].tail(5).any())
    sdiv5 = bool(sc["bear_div"].tail(5).any())
    gold5 = bool(sc["gold_buy"].tail(5).any())

    # bar-level veto — อ่านจาก sc (ใส่ไว้ตั้งแต่ score_series)
    hard_down = bool(last["hard_down"]) if "hard_down" in sc.columns else False
    bar_ret = float(last["bar_ret"]) if "bar_ret" in sc.columns else 0.0
    gold_ok = gold5 and not hard_down

    # ---------- BUY ----------
    b = []
    fresh_leading = quad_now == "Leading" and quad_prev == "Improving"
    b.append((f"Quadrant: {quad_now}" + (" (เพิ่งข้ามจาก Improving)" if fresh_leading else ""),
              quad_now == "Improving" or fresh_leading, 2))
    b.append((f"โครงสร้าง: {structure['label']}",
              structure["code"] in ("higher_low", "higher_high"), 2))
    b.append((("Pattern: " + pattern["label"]) if (pattern and pattern["tone"] == 1)
              else "ไม่มี bullish pattern", bool(pattern and pattern["tone"] == 1), 1))
    if gold_ok:
        b.append(("GOLD BUY (WT2≤-75 + RSI<30 + divergence)", True, 5))
        b.append(("WT ตัดขึ้นในโซน oversold", wt_buy3, 0))
        b.append(("Money flow เขียว", mf_green, 0))
    else:
        b.append(("WT ตัดขึ้นในโซน oversold", wt_buy3, 2))
        b.append(("Bullish divergence บน WT", bdiv5, 2))
        b.append(("Money flow เขียว", mf_green, 1))
    # แถวโปร่งใส: แต้ม 0 ไม่กระทบคะแนน แต่ผู้ใช้เห็นว่าโดน veto ด้วยเหตุอะไร
    b.append((f"แท่งล่าสุดไม่ใช่ hard down ({bar_ret:+.1%} ปิดใกล้ low)"
              if hard_down else "แท่งล่าสุดไม่ใช่ hard down",
              not hard_down, 0))
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

    # BUY / ACCUM โดนบล็อกถ้าแท่งล่าสุดพังโครงสร้างระยะสั้น
    if buy_score >= 7 and not hard_down:
        return pack("BUY", min(buy_score, 10), b)
    if sell_score >= 7:
        return pack("SELL", min(sell_score, 10), s)
    if (buy_score >= 5 and buy_score >= sell_score
            and structure["code"] in ("higher_low", "higher_high")
            and mf_green and not hard_down):
        return pack("ACCUM", buy_score, b)
    if buy_score >= 4 and buy_score >= sell_score:
        return pack("WATCH", buy_score, b)
    if sell_score >= 4:
        return pack("REDUCE", sell_score, s)
    if quad_now == "Leading" and float(last["score"]) >= 60:
        return {"grade": "HOLD", "score": 0, "checklist": []}
    return None
