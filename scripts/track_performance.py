"""
track_performance.py — วัดผลสัญญาณของ dashboard ย้อนหลัง → docs/performance.json

ใช้งาน:
    python scripts/track_performance.py                 # ราคาจาก yfinance (ใช้ cache ของ run_daily)
    python scripts/track_performance.py --offline       # ราคาจาก px_hist ใน backup/ (ทดสอบ/ไม่มีเน็ต)

หลักการ (ตกลงกันไว้ 2026-11-19):
  - หน่วยนับ = episode ไม่ใช่วัน  BUY 10 วันติด = 1 episode
    episode เริ่มวันแรกที่เกรดเปลี่ยนเข้ามา และจบเมื่อเกรดเปลี่ยนออก
  - กันสัญญาณกระพริบ: ออกแล้วกลับเข้าเกรดเดิมภายใน DEBOUNCE snapshot = episode เดิม
    (episode ของเกรดอื่นที่แทรกกลางยังนับแยกตามปกติ — BUY 1 วันกลาง WATCH ไม่หาย)
  - เข้า = ราคาปิดของแท่งที่ใช้คำนวณสัญญาณ (bar_date) · ไม่รวมค่าธรรมเนียม
  - excess = ผลตอบแทน − S&P 500 (^GSPC) ช่วงวันเดียวกัน
  - ตัวที่ไม่ใช่สินทรัพย์ลงทุนตรง (VIX, yield, FX, DXY) ไม่นับ
"""

import collections
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signals_log import LOG_PATH, BACKUP_DIR, backfill_from_backups  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "performance.json")

# ---------------- ค่าปรับจูน ----------------
DEBOUNCE = 3                    # snapshot — กลับเข้าเกรดเดิมภายในนี้ = episode เดิม
HORIZONS = (5, 10, 20, 40)      # แท่งราคาของ symbol นั้นเอง
MAIN_H = 20                     # horizon หลักที่ใช้เทียบกลยุทธ์/รายเดือน
LOOKAHEAD = 15                  # snapshot — ใช้ดู WATCH → ACCUM/BUY และกลยุทธ์ C
BENCH = "^GSPC"
BREAK_RET = 0.5                 # แท่งเดียวเปลี่ยน >50% = ข้อมูลพัง ไม่นับหน้าต่างนั้น
GRADES = ["BUY", "ACCUM", "WATCH", "HOLD", "NONE", "REDUCE", "SELL"]
NON_INVESTABLE = {"^VIX", "^TNX", "^TYX", "^IRX", "^FVX", "DX-Y.NYB"}
# ticker ที่ช่วง v1 ถูก map ผิดตัว (LSE:xxx → หุ้น US ชื่อเดียวกัน) — สัญญาณไม่ใช่ของ ETF จริง
REMAPPED_V1 = {"DFNS", "NUCL", "QNTM"}


def investable(sym: str) -> bool:
    return not (sym in NON_INVESTABLE or sym.endswith("=X") or sym in REMAPPED_V1)


# ================================================================
# ราคา
# ================================================================

def prices_online(syms: list[str]) -> dict[str, pd.Series]:
    """ราคาปิด adjusted จาก yfinance — ใน Actions จะได้จาก cache ที่ run_daily เพิ่งเขียน"""
    from fetch_data import fetch_ohlcv, drop_unclosed_bars
    from data_quality import audit
    data = drop_unclosed_bars(fetch_ohlcv(syms))
    data, _ = audit(data)
    return {s: df["Close"].astype(float) for s, df in data.items() if len(df)}


def prices_from_backups(backup_dir: str = BACKUP_DIR) -> dict[str, pd.Series]:
    """ต่อ px_hist (11 แท่งท้าย) จากทุก backup เป็น series — สำรองเมื่อดึงราคาไม่ได้

    split/การแก้ราคาย้อนหลัง: ถ้าค่าที่ซ้อนทับกับ backup ใหม่ต่างกันเป็นสัดส่วนคงที่
    (>10%) ให้ปรับสเกลของช่วงเก่าทั้งหมดตาม ไม่งั้นจะเกิดรอยต่อปลอม (เคส SOXX 3:1)
    """
    import glob
    from signals_log import _bar_date
    series: dict[str, dict] = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(backup_dir, "data_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if not d.get("watchlists"):
            continue
        seen = set()
        for w in d["watchlists"]:
            for s in w["symbols"]:
                sym = s["sym"]
                if sym in seen:
                    continue
                seen.add(sym)
                px, sd = s.get("px_hist") or [], s.get("score_dates") or []
                n = min(len(px), len(sd))
                if not n:
                    continue
                new = {pd.Timestamp(_bar_date(d["as_of"], lab)): float(p)
                       for lab, p in zip(sd[-n:], px[-n:])}
                old = series[sym]
                ratios = [new[k] / old[k] for k in new if k in old and old[k]]
                if ratios:
                    r = float(np.median(ratios))
                    if abs(r - 1) > 0.10:
                        for k in list(old):
                            old[k] *= r
                old.update(new)
    return {s: pd.Series(v).sort_index() for s, v in series.items() if v}


# ================================================================
# Episode
# ================================================================

def load_log(path: str = LOG_PATH) -> pd.DataFrame:
    log = pd.read_csv(path, dtype={"bscore": "object"})
    log = log.sort_values(["as_of", "sym", "source"]).drop_duplicates(["as_of", "sym"], keep="last")
    log["as_of"] = pd.to_datetime(log["as_of"])
    log["bar_date"] = pd.to_datetime(log["bar_date"])
    log["grade"] = log["grade"].fillna("NONE")
    dates = sorted(log["as_of"].unique())
    log["i"] = log["as_of"].map({d: i for i, d in enumerate(dates)})
    return log


def build_episodes(log: pd.DataFrame) -> pd.DataFrame:
    eps = []
    for sym, g in log.sort_values("i").groupby("sym"):
        rows = g.to_dict("records")
        last_of: dict[str, dict] = {}       # episode ล่าสุดของแต่ละเกรด (ใช้ debounce)
        cur = None
        for k, r in enumerate(rows):
            gr = r["grade"]
            if cur is not None and cur["grade"] == gr:
                cur["last_i"] = r["i"]
                continue
            prev_same = last_of.get(gr)
            if prev_same is not None and r["i"] - prev_same["last_i"] - 1 <= DEBOUNCE:
                prev_same["last_i"] = r["i"]          # กลับเข้าเกรดเดิมเร็ว = episode เดิม
                cur = prev_same
                continue
            cur = {"sym": sym, "grade": gr, "first_i": r["i"], "last_i": r["i"],
                   "start": r["as_of"], "entry_date": r["bar_date"],
                   "version": r["signal_version"], "score": r["score"],
                   "prev": rows[k - 1]["grade"] if k else None, "first_row": k == 0}
            eps.append(cur)
            last_of[gr] = cur
    E = pd.DataFrame(eps)
    # จุดจบ = snapshot ถัดจาก last_i ของ symbol นั้น (ถ้ามี) → วันออก + เกรดถัดไป
    nxt = {}
    for sym, g in log.groupby("sym"):
        gi = g.set_index("i")
        nxt[sym] = gi
    exit_date, nxt_grade, is_open = [], [], []
    for r in E.itertuples():
        gi = nxt[r.sym]
        after = gi[gi.index > r.last_i]
        if len(after):
            exit_date.append(after["bar_date"].iloc[0])
            nxt_grade.append(after["grade"].iloc[0])
            is_open.append(False)
        else:
            exit_date.append(pd.NaT)
            nxt_grade.append(None)
            is_open.append(True)
    E["exit_date"], E["next"], E["open"] = exit_date, nxt_grade, is_open
    E["days"] = E["last_i"] - E["first_i"] + 1
    # snapshot แรกของ symbol ไม่ใช่จุดเริ่มจริง (ไม่รู้ว่าเกรดนั้นเริ่มเมื่อไร)
    return E[~E["first_row"]].drop(columns="first_row").reset_index(drop=True)


# ================================================================
# ผลตอบแทน
# ================================================================

class Px:
    def __init__(self, prices: dict[str, pd.Series]):
        self.p = {s: v.dropna().sort_index() for s, v in prices.items()}
        self.bad = {s: set(v.index[v.pct_change().abs() > BREAK_RET]) for s, v in self.p.items()}
        self.bench = self.p.get(BENCH)

    def loc(self, sym, d):
        """ตำแหน่งของแท่งวันที่ d (หรือแท่งล่าสุดก่อนหน้า)"""
        p = self.p.get(sym)
        if p is None or pd.isna(d):
            return None
        k = p.index.searchsorted(pd.Timestamp(d), side="right") - 1
        return k if k >= 0 else None

    def bench_ret(self, d0, d1):
        if self.bench is None:
            return np.nan
        b0, b1 = self.bench.asof(d0), self.bench.asof(d1)
        return b1 / b0 - 1 if b0 and not pd.isna(b0) and not pd.isna(b1) else np.nan

    def window(self, sym, d0, h=None, d1=None):
        """คืน (ret, excess, mae, end_date, n_bars) จาก d0 ไป h แท่ง หรือถึงวันที่ d1"""
        k = self.loc(sym, d0)
        if k is None:
            return (np.nan,) * 3 + (None, 0)
        p = self.p[sym]
        if h is not None:
            if k + h >= len(p):
                return (np.nan,) * 3 + (None, 0)
            j = k + h
        else:
            j = self.loc(sym, d1) if d1 is not None and not pd.isna(d1) else len(p) - 1
            if j is None or j < k:
                return (np.nan,) * 3 + (None, 0)
        w = p.iloc[k:j + 1]
        if any(x in self.bad[sym] for x in w.index[1:]):
            return (np.nan,) * 3 + (None, 0)
        ret = w.iloc[-1] / w.iloc[0] - 1
        return ret, ret - self.bench_ret(w.index[0], w.index[-1]), w.min() / w.iloc[0] - 1, w.index[-1], j - k


def add_returns(E: pd.DataFrame, px: Px) -> pd.DataFrame:
    E = E.copy()
    for h in HORIZONS:
        v = [px.window(r.sym, r.entry_date, h=h) for r in E.itertuples()]
        E[f"r{h}"] = [x[0] for x in v]
        E[f"x{h}"] = [x[1] for x in v]
        E[f"mae{h}"] = [x[2] for x in v]
    # ถือจนสัญญาณเปลี่ยน (episode ที่ยังเปิด = ถึงวันนี้)
    v = [px.window(r.sym, r.entry_date, d1=r.exit_date) for r in E.itertuples()]
    E["r_exit"] = [x[0] for x in v]
    E["x_exit"] = [x[1] for x in v]
    E["bars_exit"] = [x[4] for x in v]
    return E


# ================================================================
# สรุปผล
# ================================================================

def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def stat(s_ret: pd.Series, s_xs: pd.Series, s_mae: pd.Series | None = None) -> dict:
    r, x = s_ret.dropna(), s_xs.dropna()
    out = {"n": int(len(r)),
           "avg": _r(r.mean() * 100) if len(r) else None,
           "med": _r(r.median() * 100) if len(r) else None,
           "xs": _r(x.mean() * 100) if len(x) else None,
           "xs_med": _r(x.median() * 100) if len(x) else None,
           "hit": _r((x > 0).mean() * 100, 1) if len(x) else None}
    if s_mae is not None:
        m = s_mae.dropna()
        out["mae"] = _r(m.mean() * 100) if len(m) else None
    return out


def by_grade(E: pd.DataFrame) -> list[dict]:
    out = []
    for g in GRADES:
        d = E[E.grade == g]
        row = {"grade": g, "episodes": int(len(d)), "open": int(d["open"].sum()),
               "h": {str(h): stat(d[f"r{h}"], d[f"x{h}"], d[f"mae{h}"]) for h in HORIZONS}}
        closed = d[~d["open"]]
        row["exit"] = stat(closed["r_exit"], closed["x_exit"])
        row["exit"]["bars"] = _r(closed["bars_exit"].mean(), 1) if len(closed) else None
        out.append(row)
    return out


def paths(E: pd.DataFrame, log: pd.DataFrame) -> dict:
    G = {s: g.set_index("i")["grade"] for s, g in log.groupby("sym")}
    n_snap = int(log["i"].max()) + 1

    def before(r):
        s = G[r.sym]
        return set(s[(s.index < r.first_i) & (s.index >= r.first_i - LOOKAHEAD)])

    res = {}
    for tgt in ("BUY", "ACCUM"):
        d = E[E.grade == tgt]
        c = collections.Counter()
        for r in d.itertuples():
            b = before(r)
            c["watch_before"] += "WATCH" in b
            c["accum_before"] += "ACCUM" in b
            c["neither"] += not ({"WATCH", "ACCUM"} & b)
        res[tgt.lower()] = {"n": int(len(d)), **{k: int(v) for k, v in c.items()},
                            "prev": dict(collections.Counter(d["prev"].fillna("—")).most_common())}

    d = E[(E.grade == "WATCH") & (E.first_i + LOOKAHEAD < n_snap)]      # ต้องมีอนาคตครบ 15 snapshot
    up = buy = down = 0
    lags = []
    conv = []
    for r in d.itertuples():
        s = G[r.sym]
        w = s[(s.index > r.first_i) & (s.index <= r.first_i + LOOKAHEAD)]
        u = w[w.isin(["ACCUM", "BUY"])]
        dn = w[w.isin(["REDUCE", "SELL"])]
        up += len(u) > 0
        buy += (w == "BUY").any()
        down += len(dn) > 0 and (len(u) == 0 or dn.index[0] < u.index[0])
        conv.append(len(u) > 0)
        if len(u):
            lags.append(int(u.index[0] - r.first_i))
    d = d.assign(conv=conv)
    res["watch"] = {
        "n": int(len(d)), "to_accum_or_buy": int(up), "to_buy": int(buy), "down_first": int(down),
        "median_lag": _r(np.median(lags), 1) if lags else None,
        "converted": stat(d[d.conv][f"r{MAIN_H}"], d[d.conv][f"x{MAIN_H}"], d[d.conv][f"mae{MAIN_H}"]),
        "not_converted": stat(d[~d.conv][f"r{MAIN_H}"], d[~d.conv][f"x{MAIN_H}"], d[~d.conv][f"mae{MAIN_H}"]),
    }
    return res


def strategies(E: pd.DataFrame, log: pd.DataFrame, px: Px) -> list[dict]:
    """เทียบวิธีเข้า ถือ MAIN_H แท่งนับจากวันแรกที่เริ่ม (ไม่รวมต้นทุน)"""
    G = {s: g.set_index("i") for s, g in log.groupby("sym")}
    H = MAIN_H

    def single(d):
        rows = []
        for r in d.itertuples():
            ret, xs, mae, end, _ = px.window(r.sym, r.entry_date, h=H)
            if not pd.isna(ret):
                rows.append({"dep": 1.0, "ret": ret, "xs": xs, "xs_total": xs, "mae": mae})
        return rows

    def chain(d):
        rows = []
        for r in d.itertuples():
            ret0, _, mae, end, _ = px.window(r.sym, r.entry_date, h=H)
            if pd.isna(ret0):
                continue
            g = G[r.sym]
            w = g[(g.index > r.first_i) & (g.index <= r.first_i + LOOKAHEAD)]
            legs = [r.entry_date]
            for gr in ("ACCUM", "BUY"):
                hit = w[w.grade == gr]
                if len(hit) and hit["bar_date"].iloc[0] < end:
                    legs.append(hit["bar_date"].iloc[0])
            pnl = bp = 0.0
            for d0 in legs:
                rr, xx, *_ = px.window(r.sym, d0, d1=end)
                pnl += rr / 3
                bp += (rr - xx) / 3
            dep = len(legs) / 3
            rows.append({"dep": dep, "ret": pnl / dep, "xs": (pnl - bp) / dep,
                         "xs_total": pnl - bp, "mae": mae, "legs": len(legs)})
        return rows

    W = E[E.grade == "WATCH"].copy()
    W["rising"] = W["score"] > W.apply(lambda r: _score_at(log, r.sym, r.first_i - 5), axis=1)
    defs = [
        ("A", "เข้าทั้งหมดที่ BUY", single(E[E.grade == "BUY"])),
        ("B", "เข้าทั้งหมดที่ ACCUM", single(E[E.grade == "ACCUM"])),
        ("W", "เข้าทั้งหมดที่ WATCH", single(W)),
        ("W+", "เข้าทั้งหมดที่ WATCH (คะแนนขาขึ้น 5 วัน)", single(W[W.rising])),
        ("C", "1/3 WATCH + 1/3 ACCUM + 1/3 BUY", chain(W)),
        ("C+", "เหมือน C แต่ WATCH ต้องคะแนนขาขึ้น", chain(W[W.rising])),
    ]
    out = []
    for key, name, rows in defs:
        d = pd.DataFrame(rows)
        if not len(d):
            out.append({"key": key, "name": name, "n": 0})
            continue
        item = {"key": key, "name": name, "n": int(len(d)),
                "deployed": _r(d["dep"].mean() * 100, 1),
                "ret": _r(d["ret"].mean() * 100), "xs": _r(d["xs"].mean() * 100),
                "xs_med": _r(d["xs"].median() * 100), "hit": _r((d["xs"] > 0).mean() * 100, 1),
                "xs_total": _r(d["xs_total"].mean() * 100), "mae": _r(d["mae"].mean() * 100)}
        if "legs" in d:
            item["got_2plus"] = _r((d["legs"] >= 2).mean() * 100, 1)
            item["got_3"] = _r((d["legs"] >= 3).mean() * 100, 1)
        out.append(item)
    return out


def _score_at(log, sym, i):
    m = log[(log.sym == sym) & (log.i == i)]["score"]
    return float(m.iloc[0]) if len(m) and not pd.isna(m.iloc[0]) else np.inf


def monthly(E: pd.DataFrame) -> dict:
    out = {}
    E = E.assign(month=E["start"].dt.strftime("%Y-%m"))
    for g in ("BUY", "ACCUM", "WATCH", "HOLD", "REDUCE", "SELL"):
        d = E[E.grade == g]
        rows = []
        for m, x in d.groupby("month"):
            s = stat(x[f"r{MAIN_H}"], x[f"x{MAIN_H}"])
            rows.append({"month": m, "episodes": int(len(x)), "n": s["n"], "xs": s["xs"], "hit": s["hit"]})
        out[g] = rows
    return out


def recent(E: pd.DataFrame, px: Px, n: int = 150) -> list[dict]:
    d = E[E.grade.isin(["BUY", "ACCUM", "WATCH", "SELL"])].sort_values("start", ascending=False).head(n)
    out = []
    for r in d.itertuples():
        out.append({"sym": r.sym, "grade": r.grade, "start": r.start.strftime("%Y-%m-%d"),
                    "prev": r.prev, "next": r.next, "open": bool(r.open), "days": int(r.days),
                    # ยังไม่ผ่านแท่งไหนเลย (เพิ่งเกิดวันนี้) = ยังไม่มีผล ไม่ใช่ 0%
                    "ret": _r(r.r_exit * 100) if r.bars_exit else None,
                    "xs": _r(r.x_exit * 100) if r.bars_exit else None, "bars": int(r.bars_exit),
                    "version": r.version})
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    offline = "--offline" in sys.argv
    backfill_from_backups()
    if not os.path.exists(LOG_PATH):
        raise SystemExit("ไม่มี signals_log.csv และไม่มี backup ให้เติม")
    log = load_log()
    log = log[log.sym.map(investable)]
    syms = sorted(set(log.sym)) + ([BENCH] if BENCH not in set(log.sym) else [])

    prices = {}
    if not offline:
        try:
            prices = prices_online(syms)
        except Exception as e:  # noqa
            print(f"[perf] ดึงราคาไม่สำเร็จ ({type(e).__name__}: {e}) — ใช้ราคาจาก backup แทน")
    if len(prices) < 0.8 * len(syms):
        print(f"[perf] ได้ราคา {len(prices)}/{len(syms)} — เติมด้วยราคาจาก backup")
        bk = prices_from_backups()
        for s in syms:
            prices.setdefault(s, bk.get(s, pd.Series(dtype=float)))
        price_source = "backup px_hist" if offline or not prices else "yfinance + backup"
    else:
        price_source = "yfinance (adjusted close)"
    px = Px(prices)
    if px.bench is None or not len(px.bench):
        raise SystemExit(f"ไม่มีราคา {BENCH} สำหรับ benchmark")

    E = add_returns(build_episodes(log), px)
    versions = sorted(E["version"].dropna().unique())
    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "range": [log.as_of.min().strftime("%Y-%m-%d"), log.as_of.max().strftime("%Y-%m-%d")],
        "snapshots": int(log.i.nunique()), "symbols": int(log.sym.nunique()),
        "episodes": int(len(E)),
        "params": {"debounce": DEBOUNCE, "horizons": list(HORIZONS), "main_h": MAIN_H,
                   "lookahead": LOOKAHEAD, "benchmark": BENCH, "price_source": price_source,
                   "entry": "ราคาปิดของแท่งที่ใช้คำนวณสัญญาณ"},
        "by_grade": by_grade(E),
        "by_version": {v: by_grade(E[E.version == v]) for v in versions},
        "version_counts": {v: int((E.version == v).sum()) for v in versions},
        "paths": paths(E, log),
        "strategies": strategies(E, log, px),
        "monthly": monthly(E),
        "recent": recent(E, px),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_clean(out), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    bg = {r["grade"]: r["h"][str(MAIN_H)] for r in out["by_grade"]}
    print(f"[perf] {out['snapshots']} snapshots · {out['symbols']} symbols · {out['episodes']} episodes · ราคา: {price_source}")
    print("[perf] excess %dD: " % MAIN_H + " · ".join(f"{g} {bg[g]['xs']} (n={bg[g]['n']})" for g in GRADES))
    print(f"[ok] เขียน {OUT}")


if __name__ == "__main__":
    main()
