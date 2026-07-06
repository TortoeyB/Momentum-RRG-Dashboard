"""
fetch_data.py — โหลดรายชื่อ symbol และดึง OHLCV

- อ่านธีมจาก themes.yaml
- (optional) merge symbol เพิ่มจาก watchlist.txt ที่ export จาก TradingView
  (รูปแบบ "NASDAQ:NVDA,AMEX:SMH,..." — ตัด prefix ตลาดออกอัตโนมัติ)
- ดึงราคาจาก yfinance พร้อม cache ใน data/cache (parquet)
- โหมด --demo สร้างข้อมูลจำลอง (ใช้ทดสอบ pipeline โดยไม่ต้องต่อเน็ต)
"""

import os
import re
import sys
import yaml
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

HISTORY_DAYS = 400  # ปฏิทิน ~ 270+ วันทำการ (พอสำหรับ MA200 + buffer)


def load_themes(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "themes.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for t in cfg["themes"]:
        t.setdefault("etfs", [])
        t.setdefault("stocks", [])
    return cfg


def parse_tv_watchlist(path: str) -> list[str]:
    """แปลงไฟล์ export จาก TradingView เป็น list ของ ticker
    รองรับทั้ง comma-separated และบรรทัดละตัว, ข้าม section (###...)"""
    if not os.path.exists(path):
        return []
    raw = open(path, encoding="utf-8").read()
    out = []
    for tok in re.split(r"[,\n\r]+", raw):
        tok = tok.strip()
        if not tok or tok.startswith("#"):
            continue
        sym = tok.split(":")[-1].strip().upper()   # ตัด "NASDAQ:" ฯลฯ
        if re.fullmatch(r"[A-Z0-9.\-]{1,10}", sym):
            out.append(sym)
    return sorted(set(out))


def all_symbols(cfg: dict, watchlist_path: str | None = None) -> list[str]:
    syms = {cfg.get("benchmark", "SPY")}
    for t in cfg["themes"]:
        syms.update(t["etfs"])
        syms.update(t["stocks"])
    if watchlist_path:
        extra = parse_tv_watchlist(watchlist_path)
        if extra:
            print(f"[watchlist] merge {len(extra)} symbols จาก {watchlist_path}")
            syms.update(extra)
    return sorted(syms)


# ----------------------------------------------------------------
# ดึงข้อมูลจริงผ่าน yfinance (มี cache รายวัน)
# ----------------------------------------------------------------

def fetch_ohlcv(symbols: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    today = pd.Timestamp.today().normalize()
    data = {}
    to_fetch = []
    for s in symbols:
        fp = os.path.join(CACHE, f"{s}.parquet")
        if not force and os.path.exists(fp):
            df = pd.read_parquet(fp)
            if len(df) and (today - df.index[-1]).days <= 1:
                data[s] = df
                continue
        to_fetch.append(s)

    if to_fetch:
        print(f"[fetch] ดึงราคา {len(to_fetch)} symbols จาก yfinance ...")
        raw = yf.download(to_fetch, period=f"{HISTORY_DAYS}d", interval="1d",
                          group_by="ticker", auto_adjust=True, progress=False,
                          threads=True)
        for s in to_fetch:
            try:
                df = raw[s].dropna(how="all") if len(to_fetch) > 1 else raw.dropna(how="all")
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) < 60:
                    print(f"[warn] {s}: ข้อมูลน้อยเกิน ({len(df)} แถว) — ข้าม")
                    continue
                df.to_parquet(os.path.join(CACHE, f"{s}.parquet"))
                data[s] = df
            except Exception as e:  # noqa
                print(f"[warn] {s}: ดึงไม่สำเร็จ ({e}) — ข้าม")
    return data


# ----------------------------------------------------------------
# โหมด demo — random walk มี regime เพื่อทดสอบ pipeline/dashboard
# ----------------------------------------------------------------

def synth_ohlcv(symbols: list[str], n: int = 300) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    data = {}
    for s in symbols:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        drift = rng.normal(0.0003, 0.0012)
        # สลับ regime กลางทางให้บางตัวกลับเทรนด์ จะได้เห็นครบทุก quadrant
        flip = rng.integers(0, 3)
        vol = rng.uniform(0.012, 0.028)
        rets = rng.normal(drift, vol, n)
        cut = n - rng.integers(30, 90)
        if flip == 1:
            rets[cut:] = rng.normal(-abs(drift) * 3, vol, n - cut)
        elif flip == 2:
            rets[cut:] = rng.normal(abs(drift) * 3, vol, n - cut)
        close = 100 * np.exp(np.cumsum(rets))
        o = close * (1 + rng.normal(0, vol / 3, n))
        h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
        l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
        v = rng.uniform(1e6, 5e7, n)
        data[s] = pd.DataFrame({"Open": o, "High": h, "Low": l,
                                "Close": close, "Volume": v}, index=idx)
    return data


def get_data(cfg: dict, demo: bool = False, watchlist: str | None = None,
             force: bool = False) -> dict[str, pd.DataFrame]:
    syms = all_symbols(cfg, watchlist)
    if demo:
        print(f"[demo] สร้างข้อมูลจำลอง {len(syms)} symbols")
        return synth_ohlcv(syms)
    return fetch_ohlcv(syms, force=force)


if __name__ == "__main__":
    cfg = load_themes()
    wl = os.path.join(ROOT, "watchlist.txt")
    d = get_data(cfg, demo="--demo" in sys.argv, watchlist=wl)
    print(f"ได้ข้อมูล {len(d)} symbols")
