"""
fetch_data.py — โหลดรายชื่อ symbol และดึง OHLCV

- อ่านธีมจาก themes.yaml
- (optional) merge symbol เพิ่มจาก watchlist.txt ที่ export จาก TradingView
  (รูปแบบ "NASDAQ:NVDA,AMEX:SMH,..." — ตัด prefix ตลาดออกอัตโนมัติ)
- ดึงราคาจาก yfinance พร้อม cache ใน data/cache (parquet)
- โหมด --demo สร้างข้อมูลจำลอง (ใช้ทดสอบ pipeline โดยไม่ต้องต่อเน็ต)
"""

import collections
import json
import os
import re
import sys
import time
import yaml
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

HISTORY_DAYS = 400
CACHE_TTL_HOURS = 6      # cache เก่ากว่านี้ถือว่าหมดอายุ ดึงใหม่
BATCH_SIZE = 40          # ดึงทีละกี่ symbol ต่อคำสั่ง (ก้อนใหญ่ทำให้แท่งล่าสุดหาย)
REPAIR_SLEEP = 0.15      # หน่วงระหว่างดึงซ้ำทีละตัว กัน Yahoo throttle
REPAIR_TRIES = 3         # ลองซ้ำกี่ครั้งต่อ symbol ก่อนยอมแพ้ (Yahoo throttle เป็นระยะ)

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
    "US10Y": "^TNX", "US30Y": "^TYX", "US02Y": "^IRX", "US05Y": "^FVX",
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
    "^FVX": "US 5Y Yield (x10)",
    "CL=F": "WTI Crude Oil", "BZ=F": "Brent Crude Oil", "HG=F": "Copper Futures",
    "SI=F": "Silver Futures", "GC=F": "Gold Futures", "NG=F": "Natural Gas",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
}
# รหัสที่รู้ว่า Yahoo ไม่มีแน่ๆ — ข้ามเงียบๆ ไม่ต้องพยายามดึง
# SILV/ISAG ถูกถอดออกจากลิสต์นี้แล้ว — เดิมดึงไม่ได้เพราะ prefix LSE โดนตัดทิ้ง
# ตอนนี้แมปเป็น SILV.L / ISAG.L ได้ตรงตัว
TV_SKIP = {"SET", "SET50", "TOPIX", "VNINDEX", "VN30", "CNYTHB", "3032"}

# prefix ตลาดของ TradingView → suffix ของ Yahoo
#
# เดิมโค้ดตัด prefix ทิ้งทุกตลาดยกเว้น SET: ทำให้ ticker ที่ซ้ำกันข้ามตลาด
# ถูก resolve ผิดตัวแบบเงียบๆ — เคสจริง: LSE:DFNS (VanEck Defense UCITS ETF)
# กลายเป็น DFNS เปล่าๆ ซึ่ง Yahoo คืนหุ้น T3 Defense Inc. บน NASDAQ แทน
# แล้วหุ้นตัวนั้นทำ reverse split 1:125 พอดี คะแนนทั้งแถบจึงเพี้ยน
#
# ตลาด US ไม่ต้องมี suffix — ใส่ไว้ในเซ็ตแยกเพื่อให้ "รู้จัก" ไม่ใช่ "ไม่รู้จัก"
EXCHANGE_SUFFIX = {
    "LSE": ".L", "LON": ".L",
    "XETR": ".DE", "FWB": ".DE", "GETTEX": ".DE", "TRADEGATE": ".DE",
    "EURONEXT": ".AS", "AMS": ".AS", "EPA": ".PA", "MIL": ".MI", "BME": ".MC",
    "SIX": ".SW", "OMXSTO": ".ST", "OMXCOP": ".CO", "OSL": ".OL",
    "TSX": ".TO", "TSXV": ".V",
    "TSE": ".T", "TYO": ".T",
    "HKEX": ".HK", "SEHK": ".HK",
    "SSE": ".SS", "SZSE": ".SZ",
    "KRX": ".KS", "KOSDAQ": ".KQ",
    "NSE": ".NS", "BSE": ".BO",
    "ASX": ".AX",
    "SET": ".BK",
}
US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "ARCA", "NYSEARCA", "BATS", "CBOE",
                "OTC", "US", "PINK"}
# ตลาดอนุพันธ์/ผู้ให้ข้อมูลที่ Yahoo ไม่มีสัญลักษณ์ตรงกัน — ข้ามเงียบๆ
SKIP_EXCHANGES = {"TFEX", "SPCFD", "CME", "CBOT", "COMEX", "NYMEX", "ICE"}
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
        if "/" in tok:      # กราฟอัตราส่วนของ TradingView เช่น SPCFD:SPX/AMEX:IWM
            continue        # ไม่ใช่หลักทรัพย์เดี่ยว — เดิมกลายเป็น IWM เปล่าเงียบๆ
        parts = [x.strip().upper() for x in tok.split(":")]
        sym, ex = parts[-1], (parts[0] if len(parts) > 1 else "")
        if ex in SKIP_EXCHANGES:    # ตลาดอนุพันธ์ — Yahoo ไม่มี ไม่ต้องเตือนซ้ำ
            continue
        if sym in TV_SKIP or re.fullmatch(r"S50[A-Z]\d{4}", sym):  # ดัชนี/futures ไทย ฯลฯ
            continue
        if sym in TV_TO_YAHOO:          # ดัชนี/FX/commodity — แมปตรงตัว ไม่สนตลาด
            sym = TV_TO_YAHOO[sym]
        elif ex in EXCHANGE_SUFFIX:     # ตลาดต่างประเทศ — ต้องมี suffix ไม่งั้นได้ผิดตัว
            suf = EXCHANGE_SUFFIX[ex]
            if not sym.endswith(suf):
                sym += suf
        elif ex and ex not in US_EXCHANGES:
            # prefix แปลกใหม่ที่ยังไม่รู้จัก — เตือนไว้ ดีกว่าปล่อยให้ resolve ผิดเงียบๆ
            print(f"[watchlist] ไม่รู้จักตลาด '{ex}' ของ {ex}:{sym} — "
                  f"ใช้ ticker เปล่า (อาจได้หลักทรัพย์ผิดตัว)")
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
        # ไม่เพิ่มถ้ามีลิสต์ชื่อ watchlist อยู่แล้ว (กัน tab ซ้ำต่างตัวพิมพ์)
        if syms and not any(k.lower() == "watchlist" for k in out):
            out["Watchlist"] = syms
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

def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """คัดแถวเสียโดยดูเฉพาะคอลัมน์ราคา

    เดิมใช้ .dropna() ครอบทั้ง 5 คอลัมน์ ทำให้แท่งล่าสุดหายทั้งแถวเมื่อ Volume
    ยังเป็น NaN (yfinance ปิด consolidated volume ช้ากว่า OHLC) — เป็นเหตุให้
    ข้อมูลช้าไปหนึ่งวันทำการพร้อมกันทุก symbol
    ผลพลอยได้: ดัชนี/ค่าเงินที่ไม่มี Volume จริง (^TNX, THB=X, DX-Y.NYB)
    จะไม่ถูกตัดทิ้งอีกต่อไป
    """
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    return df


# กลุ่มปฏิทินการซื้อขาย — ต้องแยกดึง ไม่งั้น yfinance ต้อง align index ข้ามปฏิทิน
# แล้วแท่งล่าสุดของกลุ่มที่ปิดทีหลังจะกลายเป็น NaN ทั้งแถว
ASIA_TICKERS = {"^N225", "^HSI", "^HSTECH", "^KS11", "^BSESN", "^NSEI",
                "000300.SS", "000905.SS"}
EU_TICKERS = {"^STOXX50E", "^STOXX"}


def _calendar_group(sym: str) -> str:
    if sym.endswith("-USD"):
        return "crypto"      # 7 วัน/สัปดาห์
    if sym.endswith("=F"):
        return "futures"     # เกือบ 24 ชม.
    if sym.endswith("=X") or sym == "DX-Y.NYB":
        return "fx"
    if sym.endswith(".BK"):
        return "th"
    if sym in ASIA_TICKERS or sym.endswith((".SS", ".SZ", ".HK", ".T", ".KS")):
        return "asia"        # ปิดก่อน US ครึ่งวัน
    if sym in EU_TICKERS or sym.endswith((".L", ".DE", ".PA", ".AS")):
        return "eu"
    return "us"


_REPAIR_OK = True   # ปิดอัตโนมัติถ้า repair ใช้การไม่ได้ในสภาพแวดล้อมนี้


def _yf_download(*args, **kw):
    """เรียก yf.download พร้อม repair=True

    repair แก้ค่าผิดพลาดที่ฝั่ง Yahoo โดยตรง — split/dividend ที่ adjust ไม่ครบ
    และ "100x error" (ราคาสลับหน่วยเพนนี/ปอนด์ ซึ่งเจอบ่อยกับ ticker .L)

    2026-08-05: เดิมดักแค่ TypeError (yfinance เก่าไม่รู้จัก kwarg) ซึ่งไม่พอ
    ถ้า scipy หายไป yfinance จะกลืน ModuleNotFoundError ไว้เองแล้วคืน
    DataFrame ว่างมาแทน ไม่ raise อะไรเลย fallback เดิมจึงไม่เคยทำงาน
    ข้อมูลถูกโยนทิ้งเงียบๆ ทั้ง 270 symbol และ pipeline ตายไป 5 วัน
    ตอนนี้เช็คผลลัพธ์ว่าง แล้วลองใหม่แบบไม่ repair ก่อนยอมแพ้
    """
    global _REPAIR_OK
    import yfinance as yf

    if _REPAIR_OK:
        try:
            out = yf.download(*args, repair=True, **kw)
            if out is not None and len(out):
                return out
            # ว่าง — อาจเป็นเพราะ repair พังเงียบ ลองใหม่แบบปกติเพื่อพิสูจน์
            plain = yf.download(*args, **kw)
            if plain is not None and len(plain):
                _REPAIR_OK = False
                print("[fetch] repair=True คืนค่าว่างแต่โหมดปกติได้ข้อมูล — "
                      "ปิด repair ทั้งรอบนี้ (เช็คว่าติดตั้ง scipy ครบหรือยัง)")
            return plain
        except TypeError as e:
            if "repair" not in str(e):
                raise
            _REPAIR_OK = False
            print("[fetch] yfinance ไม่รองรับ repair=True — ใช้โหมดปกติแทน "
                  "(แนะนำอัปเกรด yfinance)")

    return yf.download(*args, **kw)


def _download_batch(syms: list[str]) -> dict[str, pd.DataFrame]:
    """ดึงเป็นก้อนเล็ก — คืนเฉพาะตัวที่ได้ข้อมูลพอ"""
    out: dict[str, pd.DataFrame] = {}
    raw = _yf_download(syms, period=f"{HISTORY_DAYS}d", interval="1d",
                       group_by="ticker", auto_adjust=True, progress=False,
                       threads=True)
    for s in syms:
        try:
            df = raw[s].dropna(how="all") if len(syms) > 1 else raw.dropna(how="all")
            df = _tidy(df)
            if len(df) >= 60:
                out[s] = df
        except Exception:  # noqa
            pass
    return out


def _download_one(sym: str) -> pd.DataFrame | None:
    """ดึงทีละตัว — ไม่มีการ align ข้ามปฏิทิน จึงได้แท่งล่าสุดครบเสมอ"""
    df = _yf_download(sym, period=f"{HISTORY_DAYS}d", interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    df = df.droplevel(1, axis=1) if hasattr(df.columns, "levels") else df
    df = _tidy(df)
    return df if len(df) >= 60 else None


def fetch_ohlcv(symbols: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    to_fetch = []
    for s in symbols:
        fp = os.path.join(CACHE, f"{s}.parquet")
        # เดิมเทียบ (today - แท่งล่าสุด).days <= 1 ซึ่งทำให้ cache ที่มีแท่งเมื่อวาน
        # ถือว่า "สด" เสมอ แท่งของวันนี้จึงไม่เคยถูกดึง — เปลี่ยนมาใช้อายุไฟล์แทน
        if not force and os.path.exists(fp):
            age_h = (time.time() - os.path.getmtime(fp)) / 3600.0
            if age_h <= CACHE_TTL_HOURS:
                df = pd.read_parquet(fp)
                if len(df):
                    data[s] = df
                    continue
        to_fetch.append(s)

    if not to_fetch:
        return data

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for s in to_fetch:
        groups[_calendar_group(s)].append(s)
    print(f"[fetch] ดึงราคา {len(to_fetch)} symbols · "
          f"{len(groups)} กลุ่มปฏิทิน ({', '.join(f'{k}:{len(v)}' for k, v in sorted(groups.items()))})")

    for g, syms in sorted(groups.items()):
        for i in range(0, len(syms), BATCH_SIZE):
            batch = syms[i:i + BATCH_SIZE]
            try:
                data.update(_download_batch(batch))
            except Exception as e:  # noqa
                print(f"[warn] batch {g}[{i}:{i + len(batch)}] ล้มเหลว ({e})")

    # ---- ตรวจซ่อม: ตัวที่หลุด + ตัวที่แท่งล่าสุดเก่ากว่าเพื่อนในกลุ่มเดียวกัน ----
    # ใช้ mode (วันที่พบบ่อยสุด) เป็นตัวอ้างอิง ไม่ใช่ max
    # เพราะในกลุ่ม us มี ^VIX/^TNX/^TYX/^IRX ที่มีแท่งวันจันทร์ก่อนหุ้นเสมอ
    # ถ้าใช้ max หุ้นทั้งกระดานจะถูกมองว่า "ล้าหลัง" แล้วโดนดึงซ้ำทุกรอบ
    # ซึ่งทั้งช้าและทำให้ Yahoo throttle จนเกิด request ล้มเหลวแบบสุ่ม
    missing = [s for s in to_fetch if s not in data]
    stale: list[str] = []
    for g, syms in groups.items():
        lasts = {s: data[s].index[-1] for s in syms if s in data and len(data[s])}
        if not lasts:
            continue
        ref = collections.Counter(lasts.values()).most_common(1)[0][0]
        stale += [s for s, t in lasts.items() if t < ref]

    repair = missing + stale
    if repair:
        print(f"[repair] ดึงซ้ำทีละตัว {len(repair)} symbols "
              f"(หลุด {len(missing)} · ล้าหลังกลุ่ม {len(stale)}) ...")
        fixed, failed = 0, []
        for s in repair:
            before = data.get(s)
            ok = False
            for attempt in range(1, REPAIR_TRIES + 1):
                try:
                    df = _download_one(s)
                    # อย่าเอาข้อมูลที่เก่ากว่าเดิมมาทับของดี
                    if df is not None and (before is None or df.index[-1] >= before.index[-1]):
                        data[s] = df
                        if before is None or df.index[-1] > before.index[-1]:
                            fixed += 1
                        ok = True
                        break
                except Exception as e:  # noqa
                    if attempt == REPAIR_TRIES:
                        print(f"[repair] {s}: ล้มเหลว {REPAIR_TRIES} ครั้ง ({type(e).__name__}: {e})")
                time.sleep(REPAIR_SLEEP * (2 ** attempt))   # backoff
            if not ok:
                failed.append(s)
            time.sleep(REPAIR_SLEEP)
        print(f"[repair] ซ่อมได้ {fixed}/{len(repair)}")
        if failed:
            print(f"[repair] ยังซ่อมไม่ได้ {len(failed)}: {', '.join(sorted(failed))}")

    # ---- เขียน cache ----
    for s in to_fetch:
        if s in data:
            try:
                data[s].to_parquet(os.path.join(CACHE, f"{s}.parquet"))
            except Exception:  # noqa
                pass

    # ---- รายงานวันสุดท้ายต่อกลุ่ม เพื่อให้เห็นทันทีถ้ามีกลุ่มไหนค้าง ----
    for g, syms in sorted(groups.items()):
        lasts = [data[s].index[-1] for s in syms if s in data and len(data[s])]
        if lasts:
            mode = collections.Counter(lasts).most_common(1)[0][0]
            print(f"[fetch] {g:8s} {len(lasts):3d} symbols · แท่งล่าสุดส่วนใหญ่ {mode:%Y-%m-%d}")

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
