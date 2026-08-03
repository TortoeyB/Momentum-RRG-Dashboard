"""
data_quality.py — ตรวจสุขภาพ series ราคา ก่อนส่งเข้า scoring

ปัญหาที่โมดูลนี้แก้ (เคส DFNS 31 ก.ค. 2026):
    T3 Defense ทำ reverse split 1-for-125 มีผล 20 ก.ค. 2026
    แต่ราคาก่อนวันนั้นใน Yahoo ยังเป็นราคา pre-split → series มีรอยต่อ 125 เท่า
    ผลคือ 10D% = +65,804.8% และ indicator ทุกตัวที่มองย้อนข้าม 20 ก.ค.
    (MA200, MA50, HMA, ADX, WaveTrend, money flow) กลายเป็นขยะ
    โดยที่ pipeline ไม่รู้ตัวเลย แล้วยังโผล่เป็น setup อันดับ 1 ของวัน

วิธีทำงาน (3 ชั้น เรียงจากถูกไปแพง):
    1. scan รอยต่อจากราคาอย่างเดียว — ไม่มี network cost
       หา daily ratio ที่หลุดกรอบ (>= WARN_RATIO เท่า หรือ <= 1/WARN_RATIO)
    2. เฉพาะตัวที่เจอรอยต่อ ค่อยยิง yf.Ticker(sym).splits ไปถาม corporate action
       (ปกติ 0–3 ตัวจาก ~260 symbols จึงไม่กระทบเวลารัน)
    3. ถ้ารอยต่อตรงกับ ratio ของ split จริง → back-adjust ย้อนหลังให้เอง
       ถ้าไม่ตรง หรือไม่มี split รองรับ → mark เป็น bad ไม่ต้องเดา

สถานะที่คืนออกไป:
    ok    — ใช้ได้ตามปกติ (ไม่ถูกใส่ลง data.json เพื่อไม่ให้ไฟล์บวม)
    warn  — ซ่อมแล้วหรือมีจุดน่าสงสัย แต่ยังให้คะแนนได้ · หน้าเว็บขึ้นป้ายเตือน
    bad   — ประวัติเชื่อไม่ได้ · run_daily จะกดสัญญาณเป็น HOLD 0/10
"""

import numpy as np
import pandas as pd

# ---------------- ค่าปรับจูน (แก้ที่เดียว มีผลทั้งระบบ) ----------------

WARN_RATIO = 2.0        # แท่งเดียวเปลี่ยน >= 2 เท่า = ควรไปถาม corporate action
JUMP_RATIO = 4.0        # >= 4 เท่า และไม่มี split รองรับ = ข้อมูลพัง ไม่ใช่ของจริง
SPLIT_TOL = 0.15        # ยอมคลาด 15% ตอนจับคู่รอยต่อกับ ratio ของ split
MIN_BARS_AFTER_SPLIT = 30   # ต้องมีแท่งหลัง split อย่างน้อยเท่านี้ ถึงเชื่อคะแนน
SCAN_BARS = 260         # ย้อนดูรอยต่อกี่แท่ง (ครอบคลุม lookback ยาวสุด = MA200)


# ---------------- ชั้น 1: หารอยต่อจากราคา (ไม่ใช้เน็ต) ----------------

def _price_jumps(df: pd.DataFrame, thresh: float = WARN_RATIO) -> list[tuple]:
    """คืน [(วันที่, ratio)] ของแท่งที่ราคากระโดดเกินกรอบ

    ใช้ Close ต่อ Close เพราะ split ทำให้ทั้ง OHLC ขยับพร้อมกัน
    ratio > 1 = กระโดดขึ้น (ลายเซ็นของ reverse split ที่ไม่ได้ adjust)
    ratio < 1 = กระโดดลง (ลายเซ็นของ forward split ที่ไม่ได้ adjust)
    """
    px = df["Close"].tail(SCAN_BARS)
    if len(px) < 2:
        return []
    r = (px / px.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    hit = r[(r >= thresh) | (r <= 1.0 / thresh)]
    return [(d, float(v)) for d, v in hit.items()]


# ---------------- ชั้น 2: ถาม corporate action ----------------

def _norm_index(idx) -> pd.DatetimeIndex:
    """ทำให้ index เป็น naive datetime เสมอ — yfinance คืน tz-aware บ้างไม่ aware บ้าง
    ถ้าไม่ normalize การเทียบวันที่ระหว่าง splits กับ OHLC จะ raise TypeError"""
    out = pd.to_datetime(idx)
    if getattr(out, "tz", None) is not None:
        out = out.tz_convert(None)
    return out.normalize()


def fetch_splits(sym: str) -> pd.Series:
    """ประวัติ split ของ symbol — Series(index=วันที่, value=ratio)

    ratio จาก Yahoo: forward 2-for-1 → 2.0 · reverse 1-for-125 → 0.008
    คืน Series ว่างถ้าดึงไม่ได้ (จะได้ไม่ทำ pipeline ทั้งตัวล้ม)
    """
    try:
        import yfinance as yf
        s = yf.Ticker(sym).splits
    except Exception as e:  # noqa
        print(f"[quality] {sym}: ดึง split ไม่ได้ ({type(e).__name__}) — ข้ามการ back-adjust")
        return pd.Series(dtype=float)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if not len(s):
        return pd.Series(dtype=float)
    s.index = _norm_index(s.index)
    return s.sort_index()


def _match_splits(df: pd.DataFrame, splits: pd.Series) -> list[dict]:
    """จับคู่รอยต่อของราคากับ split จริง

    ถ้า series ถูก adjust มาแล้ว จะ "ไม่มี" รอยต่อที่วัน split → ไม่คืนอะไร
    ถ้ายังไม่ adjust จะเห็นรอยต่อ ≈ 1/ratio (reverse 1:125 → ratio .008 → jump ×125)
    """
    out = []
    px = df["Close"]
    idx = _norm_index(px.index)
    for dt, ratio in splits.items():
        expected = 1.0 / ratio          # รอยต่อที่ "ควรเห็น" ถ้ายังไม่ adjust
        if abs(expected - 1.0) < 0.3:   # split จิ๊บจ๊อย ไม่พอทำให้คะแนนเพี้ยน
            continue
        pos = int(idx.searchsorted(dt))
        if pos <= 0 or pos >= len(px):
            continue
        prev, cur = float(px.iloc[pos - 1]), float(px.iloc[pos])
        if prev <= 0:
            continue
        jump = cur / prev
        if abs(jump / expected - 1.0) <= SPLIT_TOL:
            out.append({"pos": pos, "date": px.index[pos], "ratio": float(ratio),
                        "jump": jump})
    return out


# ---------------- ชั้น 3: back-adjust ----------------

def backadjust(df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """คูณราคาก่อนวัน split ด้วย 1/ratio ให้ต่อเนื่องกับราคาหลัง split

    reverse 1:125 → ratio 0.008 → ราคาเก่า × 125 · volume เก่า × 0.008
    ทำจากอีเวนต์เก่าสุดไปใหม่สุด เผื่อมี split ซ้อนกันหลายรอบ
    """
    df = df.copy()
    for ev in sorted(events, key=lambda e: e["pos"]):
        m = df.index < ev["date"]
        if not m.any():
            continue
        k = 1.0 / ev["ratio"]
        for c in ("Open", "High", "Low", "Close"):
            df.loc[m, c] = df.loc[m, c] * k
        df.loc[m, "Volume"] = df.loc[m, "Volume"] * ev["ratio"]
    return df


def _fmt_split(ratio: float) -> str:
    if ratio < 1:
        return f"reverse split 1:{round(1.0 / ratio):g}"
    return f"split {round(ratio):g}:1"


# ---------------- ตัวเรียกหลัก ----------------

def audit_symbol(sym: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """ตรวจ+ซ่อม symbol เดียว — คืน (df ที่ซ่อมแล้ว, quality dict)"""
    q = {"status": "ok", "reasons": []}
    jumps = _price_jumps(df)
    if not jumps:
        return df, q

    splits = fetch_splits(sym)
    events = _match_splits(df, splits) if len(splits) else []

    if events:
        df = backadjust(df, events)
        for ev in events:
            n_after = int((df.index >= ev["date"]).sum())
            tag = _fmt_split(ev["ratio"])
            q["reasons"].append(f"{tag} เมื่อ {pd.Timestamp(ev['date']):%Y-%m-%d} — "
                                f"ราคาเดิมไม่ถูก adjust · ซ่อมย้อนหลังแล้ว")
            q["status"] = "warn"
            q["bars_since_split"] = n_after
            if n_after < MIN_BARS_AFTER_SPLIT:
                q["status"] = "bad"
                q["reasons"].append(f"มีเพียง {n_after} แท่งหลัง split "
                                    f"(ต้องการ ≥{MIN_BARS_AFTER_SPLIT}) — "
                                    f"MA/HMA/WT ยังพาดข้ามรอยต่อ")

    # รอยต่อที่เหลืออยู่หลังซ่อม = ไม่มี corporate action รองรับ
    left = _price_jumps(df, JUMP_RATIO)
    if left:
        d, v = left[-1]
        q["status"] = "bad"
        q["reasons"].append(f"ราคากระโดด ×{v:.1f} เมื่อ {pd.Timestamp(d):%Y-%m-%d} "
                            f"โดยไม่มี split รองรับ — ข้อมูลน่าจะผิด")
    elif q["status"] == "ok":
        d, v = max(jumps, key=lambda x: abs(np.log(x[1])))
        q["status"] = "warn"
        q["reasons"].append(f"ราคาเปลี่ยน ×{v:.2f} ในวันเดียว เมื่อ "
                            f"{pd.Timestamp(d):%Y-%m-%d} — ตรวจแล้วไม่ใช่ split "
                            f"(อาจเป็นของจริงในหุ้น float บาง)")
    return df, q


def audit(data: dict[str, pd.DataFrame], demo: bool = False
          ) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """ตรวจทั้งชุด — คืน (data ที่ซ่อมแล้ว, {sym: quality} เฉพาะตัวที่ไม่ ok)"""
    if demo:
        return data, {}
    fixed: dict[str, pd.DataFrame] = {}
    quality: dict[str, dict] = {}
    for sym, df in data.items():
        try:
            df2, q = audit_symbol(sym, df)
        except Exception as e:  # noqa
            print(f"[quality] {sym}: ตรวจไม่สำเร็จ ({type(e).__name__}: {e})")
            fixed[sym] = df
            continue
        fixed[sym] = df2
        if q["status"] != "ok":
            quality[sym] = q

    if quality:
        n_bad = sum(1 for q in quality.values() if q["status"] == "bad")
        print(f"[quality] พบปัญหา {len(quality)} symbols (bad {n_bad} · warn "
              f"{len(quality) - n_bad})")
        for sym, q in sorted(quality.items()):
            for r in q["reasons"]:
                print(f"[quality] {q['status'].upper():4s} {sym}: {r}")
    else:
        print("[quality] ผ่านทุก symbol — ไม่พบรอยต่อราคาผิดปกติ")
    return fixed, quality
