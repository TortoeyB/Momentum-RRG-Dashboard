"""
indicators.py — คำนวณ indicator ทั้งหมดจาก OHLCV DataFrame

รวม VMC Cipher B ที่แปลงจาก Pine Script:
  - WaveTrend (WT1/WT2) + buy/sell circles
  - Money Flow (RSI+MFI wave)
  - RSI
  - Divergence detector (regular bullish/bearish บน WT2)
  - Gold buy (WT2 <= -75 + RSI < 30 + bullish divergence)
รวมถึง Hull MA 9/20, ADX(14,14), MA50/MA200 และ candlestick patterns

DataFrame ต้องมีคอลัมน์: Open, High, Low, Close, Volume (index = วันที่)
ทุกฟังก์ชันคืน Series/DataFrame ที่ align กับ index เดิม
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (ta.rma ใน Pine)"""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

def slope_up(s: pd.Series, lookback: int = 3) -> pd.Series:
    return s > s.shift(lookback)

# ----------------------------------------------------------------
# Moving averages
# ----------------------------------------------------------------

def hull(s: pd.Series, n: int) -> pd.Series:
    """Hull Moving Average"""
    half = max(int(n / 2), 1)
    sq = max(int(np.sqrt(n)), 1)
    return wma(2 * wma(s, half) - wma(s, n), sq)

# ----------------------------------------------------------------
# RSI (Wilder)
# ----------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = rma(diff.clip(lower=0), n)
    dn = rma((-diff).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

# ----------------------------------------------------------------
# ADX (14, 14) — Wilder
# ----------------------------------------------------------------

def adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> pd.DataFrame:
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = rma(tr, di_len)
    plus_di = 100 * rma(plus_dm, di_len) / atr
    minus_di = 100 * rma(minus_dm, di_len) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["adx"] = rma(dx.fillna(0), adx_len)
    out["di_plus"] = plus_di
    out["di_minus"] = minus_di
    return out

# ----------------------------------------------------------------
# VMC Cipher B — WaveTrend core
#   Pine:  ap  = hlc3
#          esa = ema(ap, chlen=9)
#          d   = ema(abs(ap-esa), chlen)
#          ci  = (ap - esa) / (0.015 * d)
#          wt1 = ema(ci, avg=12)
#          wt2 = sma(wt1, malen=3)
# ----------------------------------------------------------------

WT_OS = -53      # oversold
WT_OB = 53       # overbought
WT_GOLD = -75    # โซน gold buy

def wavetrend(df: pd.DataFrame, chlen: int = 9, avg: int = 12, malen: int = 3) -> pd.DataFrame:
    ap = (df["High"] + df["Low"] + df["Close"]) / 3.0
    esa = ema(ap, chlen)
    d = ema((ap - esa).abs(), chlen)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci.fillna(0), avg)
    wt2 = sma(wt1, malen)
    out = pd.DataFrame(index=df.index)
    out["wt1"], out["wt2"] = wt1, wt2
    out["wt_buy"] = cross_up(wt1, wt2) & (wt2 <= WT_OS)     # จุดเขียว
    out["wt_sell"] = cross_down(wt1, wt2) & (wt2 >= WT_OB)  # จุดแดง
    out["wt_cross_up"] = cross_up(wt1, wt2)
    out["wt_cross_dn"] = cross_down(wt1, wt2)
    return out

def money_flow(df: pd.DataFrame, period: int = 60, mult: float = 150.0) -> pd.Series:
    """VMC f_rsimfi — คลื่น money flow เขียว/แดง (>0 = เขียว)"""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    raw = ((df["Close"] - df["Open"]) / rng) * mult
    return sma(raw.fillna(0), period)

# ----------------------------------------------------------------
# Divergence detector (fractal pivots บน WT2)
# ----------------------------------------------------------------

def _pivots(s: pd.Series, left: int = 2, right: int = 2, low: bool = True):
    """คืน list ของ (confirm_index_pos, pivot_pos, pivot_value)
    pivot ยืนยันหลังผ่านไป `right` แท่ง"""
    v = s.values
    n = len(v)
    piv = []
    for i in range(left, n - right):
        win = v[i - left : i + right + 1]
        if np.isnan(win).any():
            continue
        if low and v[i] == win.min() and (win > v[i]).sum() >= left + right - 1:
            piv.append((i + right, i, v[i]))
        if (not low) and v[i] == win.max() and (win < v[i]).sum() >= left + right - 1:
            piv.append((i + right, i, v[i]))
    return piv

def divergences(df: pd.DataFrame, wt2: pd.Series,
                os_zone: float = -40, ob_zone: float = 40,
                max_gap: int = 40) -> pd.DataFrame:
    """Regular bullish/bearish divergence บน WT2 เทียบราคา
    bullish: ราคาทำ low ต่ำกว่า แต่ WT2 ยก low (ทั้งคู่ในโซนล่าง)
    ยิง flag ที่แท่งซึ่ง pivot ยืนยัน"""
    out = pd.DataFrame(False, index=df.index, columns=["bull_div", "bear_div"])
    lows, highs = df["Low"].values, df["High"].values

    plow = _pivots(wt2, low=True)
    for k in range(1, len(plow)):
        cf, i, v = plow[k]
        cf0, j, v0 = plow[k - 1]
        if i - j > max_gap or cf >= len(df):
            continue
        if v0 <= os_zone and v > v0 and lows[i] < lows[j]:
            out.iloc[cf, 0] = True

    phigh = _pivots(wt2, low=False)
    for k in range(1, len(phigh)):
        cf, i, v = phigh[k]
        cf0, j, v0 = phigh[k - 1]
        if i - j > max_gap or cf >= len(df):
            continue
        if v0 >= ob_zone and v < v0 and highs[i] > highs[j]:
            out.iloc[cf, 1] = True
    return out

# ----------------------------------------------------------------
# Cipher B รวมทุกสัญญาณ
# ----------------------------------------------------------------

def cipher_b(df: pd.DataFrame) -> pd.DataFrame:
    wt = wavetrend(df)
    mf = money_flow(df)
    r = rsi(df["Close"])
    div = divergences(df, wt["wt2"])
    out = pd.concat([wt, div], axis=1)
    out["mf"] = mf
    out["rsi"] = r
    # gold buy: bullish divergence + WT2 ต่ำมาก + RSI oversold (ภายใน 3 แท่ง)
    deep = (wt["wt2"] <= WT_GOLD).rolling(3, min_periods=1).max().astype(bool)
    weak_rsi = (r < 30).rolling(3, min_periods=1).max().astype(bool)
    out["gold_buy"] = div["bull_div"] & deep & weak_rsi
    return out

# ----------------------------------------------------------------
# Candlestick patterns (rule-based)
# ----------------------------------------------------------------

def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    lower = pd.concat([c, o], axis=1).min(axis=1) - l
    down5 = c.shift(1) < c.shift(6)   # บริบท: ราคาลงมาก่อน
    up5 = c.shift(1) > c.shift(6)

    out = pd.DataFrame(index=df.index)
    out["doji"] = (body <= 0.1 * rng).fillna(False)
    out["hammer"] = ((lower >= 2 * body) & (upper <= 0.35 * body.clip(lower=1e-9) + 0.1 * rng)
                     & down5).fillna(False)
    out["shooting_star"] = ((upper >= 2 * body) & (lower <= 0.35 * body.clip(lower=1e-9) + 0.1 * rng)
                            & up5).fillna(False)
    prev_red = c.shift(1) < o.shift(1)
    prev_green = c.shift(1) > o.shift(1)
    out["bull_engulf"] = (prev_red & (c > o) & (o <= c.shift(1)) & (c >= o.shift(1)) & down5).fillna(False)
    out["bear_engulf"] = (prev_green & (c < o) & (o >= c.shift(1)) & (c <= o.shift(1)) & up5).fillna(False)
    # morning / evening star (แบบเรียบง่าย 3 แท่ง)
    small_mid = body.shift(1) <= 0.3 * rng.shift(1)
    out["morning_star"] = ((c.shift(2) < o.shift(2)) & small_mid & (c > o)
                           & (c >= (o.shift(2) + c.shift(2)) / 2) & down5).fillna(False)
    out["evening_star"] = ((c.shift(2) > o.shift(2)) & small_mid & (c < o)
                           & (c <= (o.shift(2) + c.shift(2)) / 2) & up5).fillna(False)
    return out

# ----------------------------------------------------------------
# รวมทุกอย่าง — เรียกครั้งเดียวได้ DataFrame ครบ
# ----------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["ma50"] = sma(c, 50)
    out["ma200"] = sma(c, 200)
    out["hma9"] = hull(c, 9)
    out["hma20"] = hull(c, 20)
    a = adx(df)
    out = pd.concat([out, a, cipher_b(df), candle_patterns(df)], axis=1)
    return out
