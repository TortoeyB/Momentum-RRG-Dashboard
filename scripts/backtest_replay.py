"""
backtest_replay.py — สร้างสัญญาณย้อนหลังด้วย logic ปัจจุบัน แล้ว backtest → docs/backtest.json

ใช้งาน (รันผ่าน workflow "Backtest replay" ที่กดสั่งเอง — ไม่อยู่ในรอบรายวัน):
    python scripts/backtest_replay.py                    # ดึงราคา ~3 ปีจาก yfinance
    python scripts/backtest_replay.py --start 2025-01-02 # วันเริ่ม backtest (ค่าเริ่มต้น)
    python scripts/backtest_replay.py --demo             # ข้อมูลจำลอง (ทดสอบโค้ด)

ทำอะไร:
  1) ดึงราคาทุกตัวใน watchlist ย้อน ~3 ปี (เผื่อ warm-up MA200 ก่อนวันเริ่ม)
  2) คำนวณ score_series ครั้งเดียวต่อตัว — ทดสอบแล้วว่าเป็น causal ล้วน
     (ค่าวันที่ t จากข้อมูลเต็ม = ค่าจากข้อมูลที่ตัดถึง t, 180/180 จุด 2026-11-20)
  3) เดินทีละวันทำการของตลาด US: ประเมินสัญญาณด้วย eval_at() บนข้อมูลที่ "ตัดถึงวันนั้น"
     → ได้เกรดที่ระบบ v1.2 จะให้ ณ วันนั้น โดยไม่เห็นอนาคต
  4) ใช้ตัววัดผลเดิม (track_performance) + พอร์ตกติกาเดิม (paper_portfolio) โดยไม่แก้กติกา
  5) แยกผลเป็น out-of-sample (ก่อนเริ่มเก็บสัญญาณจริง 10 ก.ค. 2026 — ช่วงที่ไม่เคยใช้ออกแบบกติกา)
     กับ in-sample (10 ก.ค. 2026 → ปัจจุบัน — ช่วงที่ใช้ตั้งกติกาชุดที่ 2)

ข้อจำกัด (แสดงบนหน้าเว็บด้วย):
  - survivorship bias: universe = watchlist ปัจจุบัน ซึ่งเลือกจากตัวที่น่าสนใจวันนี้
  - เป็นสัญญาณที่ logic v1.2 "จะให้" ไม่ใช่สัญญาณที่เคยแสดงจริงในอดีต
  - ราคา adjusted ณ วันนี้ (split/ปันผลย้อนหลังถูกปรับแล้ว)
"""

import collections
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_data  # noqa: E402
from fetch_data import load_themes, load_watchlists, drop_unclosed_bars, synth_ohlcv  # noqa: E402
from data_quality import audit  # noqa: E402
from scoring import score_series, eval_at, SIGNAL_VERSION  # noqa: E402
import track_performance as tp  # noqa: E402
import paper_portfolio as pp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "backtest.json")
REPLAY_LOG = os.path.join(ROOT, "data", "signals_replay.csv")

DEFAULT_START = "2025-01-02"
LIVE_LOG_START = "2026-07-10"     # วันแรกที่มีสัญญาณจริงใน backup → แบ่ง in/out-of-sample
FETCH_DAYS = 1100                 # ~3 ปีปฏิทิน: warm-up MA200 + ช่วงทดสอบ
MIN_BARS = 220                    # ต้องมีข้อมูลถึงวันนั้นอย่างน้อยเท่านี้ (MA200 + slope) ถึงนับสัญญาณ


def arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), d)


# ================================================================
# 1) ราคา
# ================================================================

def load_prices(syms: list[str], demo: bool) -> dict[str, pd.DataFrame]:
    if demo:
        return synth_ohlcv(syms, n=750)
    fetch_data.HISTORY_DAYS = FETCH_DAYS                       # ดึงยาวกว่ารอบรายวัน
    fetch_data.CACHE = os.path.join(ROOT, "data", "cache_replay")  # ไม่ปนกับ cache รายวัน
    os.makedirs(fetch_data.CACHE, exist_ok=True)
    data = drop_unclosed_bars(fetch_data.fetch_ohlcv(syms, force=True))
    data, _ = audit(data)
    return data


# ================================================================
# 2-3) replay สัญญาณทีละวัน
# ================================================================

def replay(data: dict[str, pd.DataFrame], wl_of: dict[str, str], start: str) -> pd.DataFrame:
    cal_src = data.get(tp.BENCH)
    if cal_src is None:
        cal_src = max(data.values(), key=len)
    days = [d for d in cal_src.index if d >= pd.Timestamp(start)]
    print(f"[replay] {len(wl_of)} symbols × {len(days)} วัน ({days[0].date()} → {days[-1].date()})")
    t0 = time.time()
    rows = []
    for sym in sorted(wl_of):
        df = data.get(sym)
        if df is None or len(df) < MIN_BARS:
            continue
        try:
            sc = score_series(df)
        except Exception as e:  # noqa
            print(f"[warn] {sym}: score ไม่สำเร็จ ({e})")
            continue
        ss = sc["score"]
        idx = df.index
        for d in days:
            k = idx.searchsorted(d, side="right")       # จำนวนแท่งที่มี ณ สิ้นวัน d
            if k < MIN_BARS:
                continue
            bar = idx[k - 1]
            if (d - bar).days > 7:                      # หยุดซื้อขาย/ข้อมูลขาดนาน — ไม่นับ
                continue
            try:
                sig, struct, patt, q_now, q_prev = eval_at(sc.iloc[:k], df.iloc[:k], ss.iloc[:k])
            except Exception:  # noqa
                continue
            rows.append({"as_of": d.strftime("%Y-%m-%d"), "sym": sym,
                         "grade": sig["grade"] if sig else "NONE",
                         "bscore": sig.get("score", "") if sig else "",
                         "score": round(float(ss.iloc[k - 1]), 1), "quadrant": q_now,
                         "bar_date": bar.strftime("%Y-%m-%d"),
                         "close": round(float(df["Close"].iloc[k - 1]), 4),
                         "watchlists": wl_of[sym], "signal_version": f"{SIGNAL_VERSION}-replay",
                         "generated_at": "", "source": "replay"})
    print(f"[replay] {len(rows):,} แถว · {time.time() - t0:.0f} วินาที")
    log = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(REPLAY_LOG), exist_ok=True)
    log.to_csv(REPLAY_LOG, index=False)
    return log


def agreement(replay_log: pd.DataFrame) -> list[dict]:
    """เทียบเกรด replay กับสัญญาณจริงใน signals_log วันที่ซ้อนกัน แยกตามเวอร์ชันสัญญาณจริง"""
    if not os.path.exists(tp.LOG_PATH):
        return []
    live = pd.read_csv(tp.LOG_PATH, dtype=str)
    live = live.sort_values(["as_of", "sym", "source"]).drop_duplicates(["as_of", "sym"], keep="last")
    m = live.merge(replay_log[["as_of", "sym", "grade"]], on=["as_of", "sym"], suffixes=("_live", "_replay"))
    out = []
    for v, g in m.groupby("signal_version"):
        out.append({"version": v, "rows": int(len(g)),
                    "same": _r((g.grade_live.fillna("NONE") == g.grade_replay).mean() * 100, 1)})
    return out


# ================================================================
# 4-5) วัดผล
# ================================================================

def to_log(rows: pd.DataFrame) -> pd.DataFrame:
    """แปลงเป็นรูปแบบเดียวกับ track_performance.load_log"""
    log = rows.copy()
    log["as_of"] = pd.to_datetime(log["as_of"])
    log["bar_date"] = pd.to_datetime(log["bar_date"])
    log["grade"] = log["grade"].fillna("NONE")
    log["score"] = pd.to_numeric(log["score"], errors="coerce")
    dates = sorted(log["as_of"].unique())
    log["i"] = log["as_of"].map({d: i for i, d in enumerate(dates)})
    return log


def period_stats(pf: pd.Series, spx: pd.Series, ew: pd.Series, a: str, b: str) -> dict:
    sel = (pf.index >= a) & (pf.index <= b)
    if sel.sum() < 2:
        return {}
    p, s, e = pf[sel], spx[sel], ew[sel]
    # ฐาน = มูลค่าวันก่อนเริ่มช่วง (ถ้ามี) จะได้นับผลตั้งแต่วันแรกของช่วงจริง
    first = np.where(sel)[0][0]
    base = lambda x, full: full.iloc[first - 1] if first > 0 else x.iloc[0]  # noqa: E731
    pr = lambda x, full: (x.iloc[-1] / base(x, full) - 1) * 100  # noqa: E731
    dd = lambda x: ((x / x.cummax()) - 1).min() * 100  # noqa: E731
    return {"from": p.index[0], "to": p.index[-1], "days": int(len(p)),
            "pf": _r(pr(p, pf)), "spx": _r(pr(s, spx)), "ew": _r(pr(e, ew)),
            "pf_dd": _r(dd(p)), "spx_dd": _r(dd(s)), "ew_dd": _r(dd(e))}


def run_portfolio(log, prices, cal_bench):
    curve, trades, open_rows, pending, skipped, cal, ent, cash = pp.simulate(log, prices)
    pf = curve["pf"]
    spx = prices[tp.BENCH].reindex(cal).ffill()
    spx = spx / spx.dropna().iloc[0] * pp.START_CAPITAL
    parts = []
    for inst in sorted({pp.PROXY.get(s, s) for s in set(log.sym)}):
        s = prices.get(inst)
        if s is None:
            continue
        s = s.reindex(cal).ffill().bfill()
        parts.append(s / s.iloc[0])
    ew = pd.concat(parts, axis=1).mean(axis=1) * pp.START_CAPITAL
    show = set(cal_bench) | {cal[-1]}
    keep = [d for d in cal if d in show]
    pf_s = pd.Series([pf.loc[d.strftime("%Y-%m-%d")] for d in keep], index=[d.strftime("%Y-%m-%d") for d in keep])
    spx_s = pd.Series(spx.loc[keep].values, index=pf_s.index)
    ew_s = pd.Series(ew.loc[keep].values, index=pf_s.index)
    inv = pd.Series([curve.loc[d.strftime("%Y-%m-%d"), "invested"] for d in keep], index=pf_s.index)
    return pf_s, spx_s, ew_s, inv, trades, open_rows, skipped


def monthly_returns(pf: pd.Series, spx: pd.Series) -> list[dict]:
    idx = pd.to_datetime(pf.index)
    out = []
    prev_pf = prev_spx = None
    for m, g in pd.Series(range(len(idx)), index=idx).groupby(idx.to_period("M")):
        last = g.iloc[-1]
        p, s = pf.iloc[last], spx.iloc[last]
        if prev_pf is None:
            first = g.iloc[0]
            prev_pf, prev_spx = pf.iloc[first], spx.iloc[first]
        out.append({"month": str(m), "pf": _r((p / prev_pf - 1) * 100), "spx": _r((s / prev_spx - 1) * 100)})
        prev_pf, prev_spx = p, s
    return out


def sensitivity(log, prices, cal_bench, oos_end) -> list[dict]:
    """ถือสูงสุดกี่วัน / stop กี่ % — ดูแค่ว่ากติกาปัจจุบันอยู่ในโซนที่ผลนิ่งหรือไม่ (ไม่ใช่ไว้จูน)"""
    keep_hold, keep_stop = pp.HOLD_BARS, pp.STOP_LOSS
    out = []
    try:
        for hb in (10, 20, 30, 40, 9999):
            for st in (None, 0.08, 0.12):
                pp.HOLD_BARS, pp.STOP_LOSS = hb, st
                pf, spx, ew, inv, tr, *_ = run_portfolio(log, prices, cal_bench)
                o = period_stats(pf, spx, ew, pf.index[0], oos_end)
                i = period_stats(pf, spx, ew, LIVE_LOG_START, pf.index[-1])
                out.append({"hold": "ไม่จำกัด" if hb > 999 else hb, "stop": None if st is None else st * 100,
                            "oos": o.get("pf"), "oos_dd": o.get("pf_dd"),
                            "is": i.get("pf"), "is_dd": i.get("pf_dd"),
                            "current": hb == keep_hold and st == keep_stop})
    finally:
        pp.HOLD_BARS, pp.STOP_LOSS = keep_hold, keep_stop
    return out


def main():
    demo = "--demo" in sys.argv
    start = arg("--start", DEFAULT_START)
    cfg = load_themes()
    wl = load_watchlists(demo=demo)
    if demo and not wl:
        wl = {"Demo": ["NVDA", "MSFT", "GLD", "XLE", "COIN", "JPM", "LLY", "URA", "AAPL", "AMZN"]}
    wl_of: dict[str, str] = {}
    for name, syms in wl.items():
        for s in syms:
            wl_of[s] = wl_of[s] + "|" + name if s in wl_of else name
    wl_of = {s: w for s, w in wl_of.items() if tp.investable(s)}
    fetch_list = sorted(set(wl_of) | {pp.PROXY.get(s, s) for s in wl_of} | {tp.BENCH})

    data = load_prices(fetch_list, demo)
    if tp.BENCH not in data:
        raise SystemExit(f"ไม่มีราคา {tp.BENCH}")
    rows = replay(data, wl_of, start)
    if not len(rows):
        raise SystemExit("replay ไม่ได้สัญญาณเลย")
    agree = agreement(rows)
    log = to_log(rows)
    prices = {s: df["Close"].astype(float).dropna() for s, df in data.items() if len(df)}
    cal_bench = set(prices[tp.BENCH].index)
    oos_end = (pd.Timestamp(LIVE_LOG_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # ---- คุณภาพสัญญาณตามเกรด แยกช่วง ----
    px = tp.Px(prices)
    E = tp.add_returns(tp.build_episodes(log), px)
    grade_by = {"all": tp.by_grade(E),
                "oos": tp.by_grade(E[E.start < LIVE_LOG_START]),
                "is": tp.by_grade(E[E.start >= LIVE_LOG_START])}
    for y in sorted(E.start.dt.year.unique()):
        grade_by[str(y)] = tp.by_grade(E[E.start.dt.year == y])

    # ---- พอร์ตกติกาปัจจุบัน ----
    pf, spx, ew, inv, trades, open_rows, skipped = run_portfolio(log, prices, cal_bench)
    periods = {"all": period_stats(pf, spx, ew, pf.index[0], pf.index[-1]),
               "oos": period_stats(pf, spx, ew, pf.index[0], oos_end),
               "is": period_stats(pf, spx, ew, LIVE_LOG_START, pf.index[-1])}
    for y in sorted({d[:4] for d in pf.index}):
        periods[y] = period_stats(pf, spx, ew, f"{y}-01-01", f"{y}-12-31")
    T = pd.DataFrame(trades)
    tstats = {}
    if len(T):
        for name, sel in (("all", T.index == T.index), ("oos", T.exit < LIVE_LOG_START), ("is", T.exit >= LIVE_LOG_START)):
            t = T[sel]
            if len(t):
                w, l_ = t[t.pnl > 0], t[t.pnl <= 0]
                tstats[name] = {"n": int(len(t)), "win": _r(len(w) / len(t) * 100, 1), "avg": _r(t.ret.mean()),
                                "avg_win": _r(w.ret.mean()) if len(w) else None,
                                "avg_loss": _r(l_.ret.mean()) if len(l_) else None,
                                "pf": _r(w.pnl.sum() / -l_.pnl.sum()) if len(l_) and l_.pnl.sum() < 0 else None,
                                "bars": _r(t.bars.mean(), 1),
                                "reasons": t.reason.str.replace(r" \(.*\)$", "", regex=True).value_counts().to_dict()}

    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "demo": demo,
        "range": [pf.index[0], pf.index[-1]], "split": LIVE_LOG_START,
        "symbols": int(log.sym.nunique()), "days": int(log.i.nunique()), "episodes": int(len(E)),
        "signal_version": f"{SIGNAL_VERSION}-replay", "rules": pp.RULES_VERSION,
        "rules_detail": {"w_buy": pp.W_BUY * 100, "w_accum": pp.W_ACCUM * 100, "hold": pp.HOLD_BARS,
                         "stop": pp.STOP_LOSS * 100 if pp.STOP_LOSS else None,
                         "reduce_frac": _r(pp.REDUCE_FRAC, 3), "max_pos": pp.MAX_POS, "cost": pp.COST * 100},
        "agreement": agree,
        "periods": periods, "trade_stats": tstats,
        "grades": grade_by,
        "curve": [{"d": d, "pf": _r(pf[d], 0), "spx": _r(spx[d], 0), "ew": _r(ew[d], 0), "inv": _r(inv[d] * 100, 1)}
                  for d in pf.index],
        "monthly": monthly_returns(pf, spx),
        "sensitivity": sensitivity(log, prices, cal_bench, oos_end),
        "skipped": len(skipped),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(pp._clean(out), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    P = periods
    print(f"[bt] out-of-sample {P['oos'].get('from')} → {P['oos'].get('to')}: พอร์ต {P['oos'].get('pf')}% · "
          f"S&P {P['oos'].get('spx')}% · DD {P['oos'].get('pf_dd')}%")
    print(f"[bt] in-sample     {P['is'].get('from')} → {P['is'].get('to')}: พอร์ต {P['is'].get('pf')}% · "
          f"S&P {P['is'].get('spx')}% · DD {P['is'].get('pf_dd')}%")
    print(f"[bt] replay ตรงกับสัญญาณจริง: {agree}")
    print(f"[ok] เขียน {OUT}")


if __name__ == "__main__":
    main()
