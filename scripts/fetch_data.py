"""
fetch_data.py — โหลดรายชื่อ symbol และดึง OHLCV

- อ่านธีมจาก themes.yaml
- (optional) merge symbol เพิ่มจาก watchlist.txt ที่ export จาก TradingView
  (รูปแบบ "NASDAQ:NVDA,AMEX:SMH,..." — ตัด prefix ตลาดออกอัตโนมัติ)
- ดึงราคาจาก yfinance พร้อม cache ใน data/cache (parquet)
- โหมด --demo สร้างข้อมูลจำลอง (ใช้ทดสอบ pipeline โดยไม่ต้องต่อเน็ต)
"""

import json
import os
import re
import sys
import yaml
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

HISTORY_DAYS = 400

# รหัสดัชนี/FX/commodity ของ TradingView → ticker ของ Yahoo Finance
TV_TO_YAHOO = {
    # ดัชนี US
    "SPX": "^GSPC", "IXIC": "^IXIC", "DJI": "^DJI", "RUT": "^RUT", "VIX": "^VIX",
    # ดัชนีต่างประเทศ
    "NI225": "^N225", "HSI": "^HSI", "HSTECH": "^HSTECH", "KOSPI": "^KS11",
    "SENSEX": "^BSESN", "NIFTY": "^NSEI", "SX5E": "^STOXX50E", "SXXP": "^STOXX",
    "000300": "000300.SS", "000905": "000905.SS",
    # ค่าเงิน
    "USDTHB": "THB=X", "JPYTHB": "JPYTHB=X", "EURTHB": "EURTHB=X",
    "DXY": "DX-Y.NYB",
    # bond yield (Yahoo คูณ 10 เช่น ^TNX = 10Y yield x10)
    "US10Y": "^TNX", "US30Y": "^TYX", "US02Y": "^IRX",
    # commodity futures
    "USOIL": "CL=F", "BRENT": "BZ=F", "COPPER": "HG=F", "SILVER": "SI=F",
    "GOLD": "GC=F", "NATGAS": "NG=F",
    # crypto
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
}
# ชื่อสำเร็จรูป (ตัวที่ชื่อบน Yahoo อ่านยากหรือดึงไม่ได้)
NAME_OVERRIDES = {
    "^GSPC": "S&P 500 Index", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones Industrial",
    "^RUT": "Russell 2000", "^VIX": "CBOE Volatility Index", "^N225": "Nikkei 225",
    "^HSI": "Hang Seng Index", "^HSTECH": "Hang Seng Tech", "^KS11": "KOSPI Composite",
    "^BSESN": "BSE Sensex", "^NSEI": "Nifty 50", "^STOXX50E": "Euro Stoxx 50",
    "^STOXX": "Stoxx Europe 600", "000300.SS": "CSI 300 (จีน)", "000905.SS": "CSI 500 (จีน)",
    "THB=X": "USD/THB", "JPYTHB=X": "JPY/THB", "EURTHB=X": "EUR/THB",
    "DX-Y.NYB": "US Dollar Index (DXY)", "^TNX": "US 10Y Yield (x10)",
    "^TYX": "US 30Y Yield (x10)", "^IRX": "US 13W Yield",
    "CL=F": "WTI Crude Oil", "BZ=F": "Brent Crude Oil", "HG=F": "Copper Futures",
    "SI=F": "Silver Futures", "GC=F": "Gold Futures", "NG=F": "Natural Gas",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
}
# รหัสที่รู้ว่า Yahoo ไม่มีแน่ๆ — ข้ามเงียบๆ ไม่ต้องพยายามดึง
TV_SKIP = {"SET", "SET50", "TOPIX", "VNINDEX", "VN30", "SILV", "ISAG",
           "CNYTHB", "3032"}
  # ปฏิทิน ~ 270+ วันทำการ (พอสำหรับ MA200 + buffer)


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
        if sym in TV_SKIP or re.fullmatch(r"S50[A-Z]\d{4}", sym):  # futures ไทย ฯลฯ
            continue
        sym = TV_TO_YAHOO.get(sym, sym)
        if re.fullmatch(r"[A-Z0-9.^=\-]{1,12}", sym):
            out.append(sym)
    return sorted(set(out))


def load_watchlists(demo: bool = False) -> dict[str, list[str]]:
    """อ่านทุกไฟล์ใน watchlists/*.txt (ชื่อไฟล์ = ชื่อ tab)
    รองรับ watchlist.txt เดี่ยวแบบเก่าเป็น tab ชื่อ "Watchlist" """
    out: dict[str, list[str]] = {}
    wdir = os.path.join(ROOT, "watchlists")
    if os.path.isdir(wdir):
        for f in sorted(os.listdir(wdir)):
            if f.lower().endswith(".txt"):
                syms = parse_tv_watchlist(os.path.join(wdir, f))
                if syms:
                    out[os.path.splitext(f)[0]] = syms
    legacy = os.path.join(ROOT, "watchlist.txt")
    if os.path.exists(legacy):
        syms = parse_tv_watchlist(legacy)
        if syms:
            out.setdefault("Watchlist", syms)
    if out:
        print(f"[watchlist] พบ {len(out)} ลิสต์: {', '.join(out)}")
    return out


def all_symbols(cfg: dict, extra: list[str] | None = None) -> list[str]:
    syms = {cfg.get("benchmark", "SPY")}
    for t in cfg["themes"]:
        syms.update(t["etfs"])
        syms.update(t["stocks"])
    if extra:
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
        # retry ตัวที่หลุด (เช่น database is locked จากการดึงพร้อมกัน) ทีละตัว
        missing = [s for s in to_fetch if s not in data]
        for s in missing:
            try:
                df = yf.download(s, period=f"{HISTORY_DAYS}d", interval="1d",
                                 auto_adjust=True, progress=False, threads=False)
                df = df.droplevel(1, axis=1) if hasattr(df.columns, "levels") else df
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) >= 60:
                    df.to_parquet(os.path.join(CACHE, f"{s}.parquet"))
                    data[s] = df
                    print(f"[retry] {s}: สำเร็จรอบสอง")
            except Exception:  # noqa
                pass
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


def load_names(symbols: list[str], demo: bool = False) -> dict[str, str]:
    """ชื่อเต็มของแต่ละ ticker — cache ใน data/names.json ดึงเฉพาะตัวที่ยังไม่มี"""
    path = os.path.join(ROOT, "data", "names.json")
    names: dict[str, str] = {}
    if os.path.exists(path):
        try:
            names = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa
            names = {}
    for k, v in NAME_OVERRIDES.items():
        names.setdefault(k, v)
    missing = [s for s in symbols if not names.get(s)]
    if missing and not demo:
        import yfinance as yf
        print(f"[names] ดึงชื่อเต็ม {len(missing)} symbols ...")
        for s in missing:
            try:
                info = yf.Ticker(s).info
                names[s] = info.get("longName") or info.get("shortName") or ""
            except Exception:  # noqa
                names[s] = ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=0, sort_keys=True)
    return names


def get_data(cfg: dict, demo: bool = False, extra: list[str] | None = None,
             force: bool = False) -> dict[str, pd.DataFrame]:
    syms = all_symbols(cfg, extra)
    if demo:
        print(f"[demo] สร้างข้อมูลจำลอง {len(syms)} symbols")
        return synth_ohlcv(syms)
    return fetch_ohlcv(syms, force=force)


if __name__ == "__main__":
    cfg = load_themes()
    wl = load_watchlists()
    extra = sorted({x for v in wl.values() for x in v})
    d = get_data(cfg, demo="--demo" in sys.argv, extra=extra)
    print(f"ได้ข้อมูล {len(d)} symbols")
