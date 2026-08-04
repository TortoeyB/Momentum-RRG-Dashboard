"""
probe_yahoo.py — ตรวจว่า Yahoo ส่งอะไรกลับมาจริงๆ

รันมือจากหน้า Actions ผ่าน .github/workflows/probe.yml
ไม่แตะ data.json ไม่ commit อะไรทั้งสิ้น

ตอบคำถามเดียว: ทำไม len(df) ถึงน้อยกว่า 60 จนถูก guard โยนทิ้ง
  (A) Yahoo ไม่ส่งข้อมูลมาเลย        -> rows = 0
  (B) ส่งมาแต่ OHLC เป็น null       -> rows เยอะ แต่ NaN เยอะตาม
  (C) repair=True กินแถวทิ้ง        -> repair=False ได้เยอะกว่าชัดเจน
"""

import sys
import traceback

SYMS = sys.argv[1:] or ["AAPL", "SPY", "BTC-USD", "^GSPC"]
PERIOD = "400d"


def hr(t=""):
    print("\n" + "=" * 70)
    if t:
        print(t)
        print("=" * 70)


# ---------------------------------------------------------------- env
hr("ENVIRONMENT")
import numpy as np
import pandas as pd
import yfinance as yf

print("yfinance :", yf.__version__)
print("pandas   :", pd.__version__)
print("numpy    :", np.__version__)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


# ------------------------------------------------- 1) HTTP ดิบ
# ยิงตรงไปที่ chart endpoint เพื่อดู status code จริง
# 200 = เข้าถึงได้ (ตัดเรื่องบล็อก IP ออก) / 429 = โดน rate limit จริง
hr("1) RAW HTTP -> query1.finance.yahoo.com")
try:
    import requests
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
           "?range=1mo&interval=1d")
    r = requests.get(url, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    print("status :", r.status_code)
    print("bytes  :", len(r.content))
    if r.status_code == 200:
        j = r.json()
        res = (j.get("chart") or {}).get("result") or []
        if res:
            q = res[0]["indicators"]["quote"][0]
            ts = res[0].get("timestamp") or []
            closes = q.get("close") or []
            nulls = sum(1 for c in closes if c is None)
            print(f"timestamps : {len(ts)}")
            print(f"close      : {len(closes)} ค่า · เป็น null {nulls} ค่า")
            print(f"close 5 ตัวท้าย : {closes[-5:]}")
        else:
            print("ไม่มี result ใน payload:", str(j)[:400])
    else:
        print("body:", r.text[:400])
except Exception:
    traceback.print_exc()


# ------------------------------- 2) yf.download ทีละตัว repair ปิด/เปิด
def tidy_like_pipeline(df):
    """เลียนแบบ _tidy() ใน fetch_data.py เป๊ะๆ"""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def probe_one(sym, repair):
    tag = f"{sym} repair={repair}"
    try:
        df = yf.download(sym, period=PERIOD, interval="1d",
                         auto_adjust=True, progress=False,
                         threads=False, repair=repair)
    except Exception as e:
        print(f"[{tag}] EXCEPTION {type(e).__name__}: {e}")
        return
    if df is None or not len(df):
        print(f"[{tag}] ว่างเปล่า — Yahoo ไม่ส่งแถวมาเลย")
        return

    if hasattr(df.columns, "levels"):
        df = df.droplevel(1, axis=1)

    raw_rows = len(df)
    nan_counts = {c: int(df[c].isna().sum())
                  for c in df.columns if c in
                  ("Open", "High", "Low", "Close", "Volume")}
    kept = len(tidy_like_pipeline(df.copy()))

    print(f"[{tag}] แถวดิบ {raw_rows} · หลัง _tidy เหลือ {kept} "
          f"· guard(>=60) {'ผ่าน' if kept >= 60 else 'ตก <-- ตรงนี้'}")
    print(f"    NaN รายคอลัมน์: {nan_counts}")
    print(f"    วันที่ล่าสุด : {df.index[-1]}")
    print("    3 แท่งท้าย:")
    print(df.tail(3).to_string().replace("\n", "\n    "))


hr("2) yf.download ทีละตัว — เทียบ repair=False vs True")
for s in SYMS:
    print(f"\n--- {s} ---")
    probe_one(s, repair=False)
    probe_one(s, repair=True)


# ------------------------------------------- 3) โหมด batch แบบ pipeline จริง
hr("3) yf.download แบบก้อน (group_by='ticker') เหมือน _download_batch")
batch = [s for s in SYMS if not s.startswith("^") and "-" not in s][:3]
if len(batch) >= 2:
    try:
        raw = yf.download(batch, period=PERIOD, interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True, repair=True)
        print("shape:", raw.shape)
        for s in batch:
            try:
                sub = raw[s].dropna(how="all")
                kept = len(tidy_like_pipeline(sub.copy()))
                print(f"  {s}: ดิบ {len(sub)} · หลัง _tidy {kept} "
                      f"· {'ผ่าน' if kept >= 60 else 'ตก'}")
            except Exception as e:
                print(f"  {s}: ดึงจาก frame ไม่ได้ ({type(e).__name__}: {e})")
    except Exception:
        traceback.print_exc()
else:
    print("ข้าม — symbol ไม่พอ")

hr("จบ")
