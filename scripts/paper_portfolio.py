"""
paper_portfolio.py — พอร์ตจำลองที่เทรดตามสัญญาณ dashboard → docs/portfolio.json

ใช้งาน:
    python scripts/paper_portfolio.py              # ราคาจาก yfinance (ใช้ cache ของ run_daily)
    python scripts/paper_portfolio.py --offline    # ราคาจาก px_hist ใน backup/ (ทดสอบ)

กติกา baseline (ตกลงกันไว้ 2026-11-19 — "ตั้งครั้งเดียว ไม่จูนตามผล"):
  เข้า   BUY = ไม้เต็ม (10% ของพอร์ต) · ACCUM = ครึ่งไม้ (5%)
         ถือ ACCUM อยู่แล้วเจอ BUY → เติมอีก 5% ให้เต็มไม้  (ทยอยซื้อ)
         เข้าเฉพาะวันแรกของ episode (ไม่ไล่ซื้อสัญญาณเดิมซ้ำ) · WATCH ไม่เข้า
  ออก   ครบ HOLD_BARS แท่ง หรือ symbol ต้นทางขึ้น SELL/REDUCE — แล้วแต่อย่างไหนก่อน
  ขนาด  ถือพร้อมกันได้ MAX_POS ตัว · เงินที่เหลือ = เงินสด 0% · ไม่ใช้ margin
  ราคา  ส่งคำสั่ง = ราคาปิด "แท่งถัดไป" หลังแท่งที่เกิดสัญญาณ (สัญญาณคำนวณหลังตลาดปิด
         จึงซื้อที่ราคาปิดของแท่งนั้นเองไม่ได้จริง) · ค่าธรรมเนียม COST ต่อฝั่ง
  ดัชนี/futures ซื้อตรงไม่ได้ → ใช้ ETF ตัวแทน (PROXY) แต่สัญญาณยังมาจากตัวต้นทาง

ไม่ใช่คำแนะนำการลงทุน — ใช้ดูพฤติกรรมของระบบสัญญาณเท่านั้น
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signals_log import backfill_from_backups, LOG_PATH  # noqa: E402
from track_performance import (load_log, build_episodes, investable,  # noqa: E402
                               prices_online, prices_from_backups, BENCH)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "portfolio.json")

# ---------------- กติกา (แก้ = ต้องบอกในหน้าเว็บ เพราะผลย้อนหลังจะเปลี่ยนทั้งหมด) ----------------
START_CAPITAL = 1_000_000.0
MAX_POS = 10
W_BUY = 0.10
W_ACCUM = 0.05
HOLD_BARS = 20
EXIT_GRADES = {"SELL", "REDUCE"}
COST = 0.0015
RULES_VERSION = "baseline-1"

PROXY = {
    "^GSPC": "SPY", "^IXIC": "QQQ", "NDX": "QQQ", "^NDX": "QQQ", "^DJI": "DIA", "^RUT": "IWM",
    "^NSEI": "INDA", "^BSESN": "INDA", "^HSI": "EWH", "^HSTECH": "KWEB", "^N225": "EWJ",
    "^KS11": "EWY", "000300.SS": "ASHR", "000905.SS": "ASHR",
    "^STOXX50E": "FEZ", "^STOXX": "VGK",
    "SI=F": "SLV", "GC=F": "GLD", "CL=F": "USO", "BZ=F": "BNO", "HG=F": "CPER", "NG=F": "UNG",
}


def _r(x, d=2):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), d)


class Book:
    """พอร์ตแบบถือเป็นจำนวนหน่วย — มูลค่าตามราคาปิดล่าสุด (asof) ของแต่ละตัว"""

    def __init__(self, prices):
        self.p = prices
        self.cash = START_CAPITAL
        self.pos: dict[str, dict] = {}          # key = instrument
        self.trades: list[dict] = []

    def px(self, inst, d):
        s = self.p.get(inst)
        if s is None or not len(s):
            return np.nan
        v = s.asof(d)
        return float(v) if not pd.isna(v) else np.nan

    def equity(self, d):
        return self.cash + sum(p["units"] * self.px(k, d) for k, p in self.pos.items()
                               if not math.isnan(self.px(k, d)))

    def buy(self, inst, d, value, meta):
        price = self.px(inst, d)
        value = min(value, self.cash / (1 + COST))
        if math.isnan(price) or value <= 0:
            return False
        units = value / price
        self.cash -= value * (1 + COST)
        p = self.pos.get(inst)
        if p:                                    # เติมไม้ (ACCUM → BUY)
            p["grade"] = "ACCUM→BUY"
            p["cost"] += value * (1 + COST)
            p["units"] += units
            p["adds"].append({"date": d.strftime("%Y-%m-%d"), "px": _r(price, 4), "grade": meta["grade"]})
        else:
            self.pos[inst] = {"units": units, "cost": value * (1 + COST), "entry": d, "entry_px": price,
                              "sym": meta["sym"], "grade": meta["grade"], "bars": 0,
                              "adds": [{"date": d.strftime("%Y-%m-%d"), "px": _r(price, 4), "grade": meta["grade"]}]}
        return True

    def sell(self, inst, d, reason):
        p = self.pos.pop(inst)
        price = self.px(inst, d)
        proceeds = p["units"] * price * (1 - COST)
        self.cash += proceeds
        self.trades.append(self._trade_row(inst, p, d, price, proceeds, reason, open_=False))

    def _trade_row(self, inst, p, d, price, value, reason, open_):
        return {"sym": p["sym"], "inst": inst, "grade": p["grade"],
                "entry": p["entry"].strftime("%Y-%m-%d"), "entry_px": _r(p["entry_px"], 4),
                "exit": d.strftime("%Y-%m-%d"), "exit_px": _r(price, 4),
                "ret": _r((value / p["cost"] - 1) * 100), "bars": int(p["bars"]),
                "weight": _r(p["cost"] / START_CAPITAL * 100, 1), "pnl": _r(value - p["cost"], 0),
                "reason": reason, "open": open_, "adds": p["adds"]}


def next_bar(series: pd.Series, after) -> pd.Timestamp | None:
    k = series.index.searchsorted(pd.Timestamp(after), side="right")
    return series.index[k] if k < len(series) else None


def simulate(log: pd.DataFrame, prices: dict[str, pd.Series]):
    E = build_episodes(log)
    inst_of = lambda s: PROXY.get(s, s)  # noqa: E731
    # สัญญาณเข้า: วันแรกของ episode BUY / ACCUM
    ent = E[E.grade.isin(["BUY", "ACCUM"])].copy()
    ent["inst"] = ent.sym.map(inst_of)
    ent = ent[ent.inst.map(lambda i: i in prices and len(prices[i]) > 0)]
    # เกรดของ symbol ต้นทางทุกวัน (ใช้หาจุดออกตามสัญญาณ)
    grade_at = {(r.sym, r.as_of): r.grade for r in log.itertuples()}
    snaps = sorted(log.as_of.unique())
    bar_of = {(r.sym, r.as_of): r.bar_date for r in log.itertuples()}

    bench = prices[BENCH]
    start = pd.Timestamp(snaps[0])
    end = max(s.index.max() for s in prices.values() if len(s))
    cal = sorted(set(d for s in prices.values() for d in s.index if start <= d <= end))
    cal = pd.DatetimeIndex(cal)

    book = Book(prices)
    pending: list[dict] = []        # {"date", "side", "inst", ...}
    snap_set = set(pd.Timestamp(x) for x in snaps)
    ent_by_day = {d: g for d, g in ent.groupby("start")}
    curve, skipped = [], []

    for d in cal:
        # ---- 1) ออกเพราะถือครบ HOLD_BARS (นับแท่งของตัวที่ถือจริง) ----
        for inst, p in list(book.pos.items()):
            s = prices[inst]
            if d in s.index and d > p["entry"]:
                p["bars"] += 1
                if p["bars"] >= HOLD_BARS:
                    book.sell(inst, d, f"ครบ {HOLD_BARS} วัน")
        # ---- 2) คำสั่งที่ถึงวันส่ง (ขายก่อน ซื้อทีหลัง) ----
        due = [o for o in pending if o["date"] == d]
        pending = [o for o in pending if o["date"] != d]
        for o in sorted(due, key=lambda o: o["side"] != "sell"):
            if o["side"] == "sell" and o["inst"] in book.pos:
                book.sell(o["inst"], d, o["reason"])
            elif o["side"] == "buy":
                held = o["inst"] in book.pos
                if held and not o.get("topup"):
                    continue
                if not held and len(book.pos) >= MAX_POS:
                    skipped.append({**o["meta"], "date": d.strftime("%Y-%m-%d"), "why": "พอร์ตเต็ม"})
                    continue
                if not book.buy(o["inst"], d, o["weight"] * book.equity(d), o["meta"]):
                    skipped.append({**o["meta"], "date": d.strftime("%Y-%m-%d"), "why": "เงินสดไม่พอ"})
        # ---- 3) mark-to-market ----
        eq = book.equity(d)
        curve.append({"d": d.strftime("%Y-%m-%d"), "pf": eq,
                      "invested": (eq - book.cash) / eq if eq else 0, "n": len(book.pos)})
        # ---- 4) อ่านสัญญาณของวันนี้ (รู้หลังตลาดปิด) → สร้างคำสั่งไปส่งแท่งถัดไป ----
        if d not in snap_set:
            continue
        # 4a ออกตามสัญญาณ
        for inst, p in list(book.pos.items()):
            g = grade_at.get((p["sym"], d))
            if g in EXIT_GRADES and not any(o["inst"] == inst and o["side"] == "sell" for o in pending):
                nb = next_bar(prices[inst], bar_of.get((p["sym"], d), d))
                if nb is not None:
                    pending.append({"date": nb, "side": "sell", "inst": inst, "reason": f"สัญญาณ {g}"})
        # 4b เข้าใหม่ — BUY ก่อน ACCUM, คะแนนสูงก่อน
        if d in ent_by_day:
            todo = ent_by_day[d].assign(pri=lambda x: (x.grade != "BUY").astype(int))
            for r in todo.sort_values(["pri", "score"], ascending=[True, False]).itertuples():
                held = book.pos.get(r.inst)
                topup = bool(held and held["grade"] == "ACCUM" and r.grade == "BUY")
                if held and not topup:
                    continue
                if any(o["inst"] == r.inst and o["side"] == "buy" for o in pending):
                    continue
                nb = next_bar(prices[r.inst], r.entry_date)
                if nb is None:
                    continue
                w = W_BUY if r.grade == "BUY" else W_ACCUM
                if topup:
                    w = W_BUY - W_ACCUM
                pending.append({"date": nb, "side": "buy", "inst": r.inst, "weight": w, "topup": topup,
                                "meta": {"sym": r.sym, "grade": r.grade}})

    last = cal[-1]
    open_rows = []
    for inst, p in book.pos.items():
        price = book.px(inst, last)
        open_rows.append(book._trade_row(inst, p, last, price, p["units"] * price * (1 - COST), "ยังถืออยู่", True))
    pending_rows = [{"date": o["date"].strftime("%Y-%m-%d"), "side": o["side"], "inst": o["inst"],
                     **({"sym": o["meta"]["sym"], "grade": o["meta"]["grade"]} if "meta" in o else {"reason": o["reason"]})}
                    for o in pending]
    return pd.DataFrame(curve).set_index("d"), book.trades, open_rows, pending_rows, skipped, cal, ent


def stats(curve: pd.Series, trades: list[dict], invested: pd.Series) -> dict:
    ret = curve.iloc[-1] / curve.iloc[0] - 1
    dd = curve / curve.cummax() - 1
    t = pd.DataFrame(trades)
    out = {"ret": _r(ret * 100), "max_dd": _r(dd.min() * 100),
           "max_dd_date": dd.idxmin() if len(dd) else None,
           "exposure": _r(invested.mean() * 100, 1), "trades": int(len(t))}
    if len(t):
        wins, losses = t[t.ret > 0], t[t.ret <= 0]
        out.update({"win_rate": _r(len(wins) / len(t) * 100, 1), "avg_trade": _r(t.ret.mean()),
                    "avg_win": _r(wins.ret.mean()) if len(wins) else None,
                    "avg_loss": _r(losses.ret.mean()) if len(losses) else None,
                    "profit_factor": _r(wins.pnl.sum() / -losses.pnl.sum()) if len(losses) and losses.pnl.sum() < 0 else None,
                    "avg_bars": _r(t.bars.mean(), 1),
                    "best": t.loc[t.ret.idxmax(), ["inst", "ret"]].to_dict(),
                    "worst": t.loc[t.ret.idxmin(), ["inst", "ret"]].to_dict()})
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
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def main():
    offline = "--offline" in sys.argv
    backfill_from_backups()
    if not os.path.exists(LOG_PATH):
        raise SystemExit("ไม่มี signals_log.csv")
    log = load_log()
    log = log[log.sym.map(investable)]
    srcs = sorted(set(log.sym))
    insts = sorted({PROXY.get(s, s) for s in srcs} | {BENCH})

    prices = {}
    if not offline:
        try:
            prices = prices_online(sorted(set(insts) | set(srcs)))
        except Exception as e:  # noqa
            print(f"[pf] ดึงราคาไม่สำเร็จ ({type(e).__name__}: {e}) — ใช้ราคาจาก backup")
    bk = None
    proxy_missing = []
    for inst in insts:
        if inst in prices and len(prices[inst]):
            continue
        if bk is None:
            bk = prices_from_backups()
        src = next((s for s in srcs if PROXY.get(s) == inst), inst)
        if inst in bk:
            prices[inst] = bk[inst]
        elif src in bk:                     # ไม่มีราคา ETF ตัวแทน → ใช้ราคาตัวต้นทาง (บอกในหน้าเว็บ)
            prices[inst] = bk[src]
            proxy_missing.append(inst)
    prices = {k: v.dropna().sort_index() for k, v in prices.items() if v is not None and len(v.dropna())}
    if BENCH not in prices:
        raise SystemExit(f"ไม่มีราคา {BENCH}")

    curve, trades, open_rows, pending, skipped, cal, ent = simulate(log, prices)
    pf = curve["pf"]

    # benchmark: S&P 500 และ equal-weight ทุกตัวที่ซื้อได้ ณ วันแรก (ซื้อแล้วถือ)
    spx = prices[BENCH].reindex(cal).ffill()
    spx = spx / spx.dropna().iloc[0] * START_CAPITAL
    ew_parts = []
    for inst in sorted({PROXY.get(s, s) for s in srcs}):
        s = prices.get(inst)
        if s is None:
            continue
        s = s.reindex(cal).ffill()
        if pd.isna(s.iloc[0]):
            continue
        ew_parts.append(s / s.iloc[0])
    ew = pd.concat(ew_parts, axis=1).mean(axis=1) * START_CAPITAL if ew_parts else spx * np.nan

    # กราฟแสดงเฉพาะวันทำการของ S&P (+ วันสุดท้าย) — วันเสาร์อาทิตย์ของ crypto ทำให้เส้นเป็นขั้นบันได
    show = set(prices[BENCH].index) | {cal[-1]}
    curve_out = [{"d": d.strftime("%Y-%m-%d"), "pf": _r(v, 0), "spx": _r(spx.loc[d], 0), "ew": _r(ew.loc[d], 0),
                  "inv": _r(curve.loc[d.strftime("%Y-%m-%d"), "invested"] * 100, 1),
                  "n": int(curve.loc[d.strftime("%Y-%m-%d"), "n"])}
                 for d, v in zip(cal, pf.values) if d in show]

    # ราคาของตัวที่เคยเทรด — ใช้วาดจุดซื้อ/ขาย
    traded = sorted({t["inst"] for t in trades} | {t["inst"] for t in open_rows})
    px_out = {}
    for inst in traded:
        s = prices[inst]
        s = s[(s.index >= cal[0]) & (s.index <= cal[-1])]
        px_out[inst] = {"d": [x.strftime("%Y-%m-%d") for x in s.index], "c": [_r(v, 4) for v in s.values]}

    pf_stats = stats(pf, trades, curve["invested"])
    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "range": [cal[0].strftime("%Y-%m-%d"), cal[-1].strftime("%Y-%m-%d")],
        "rules": {"version": RULES_VERSION, "capital": START_CAPITAL, "max_pos": MAX_POS,
                  "w_buy": W_BUY * 100, "w_accum": W_ACCUM * 100, "hold_bars": HOLD_BARS,
                  "exit_grades": sorted(EXIT_GRADES), "cost": COST * 100,
                  "execution": "ราคาปิดแท่งถัดไปหลังเกิดสัญญาณ", "proxy": PROXY,
                  "proxy_missing": sorted(set(proxy_missing)),
                  "price_source": "backup px_hist" if offline else "yfinance (adjusted close)"},
        "stats": {"portfolio": pf_stats,
                  "spx": {"ret": _r((spx.iloc[-1] / spx.iloc[0] - 1) * 100),
                          "max_dd": _r((spx / spx.cummax() - 1).min() * 100)},
                  "ew": {"ret": _r((ew.iloc[-1] / ew.iloc[0] - 1) * 100),
                         "max_dd": _r((ew / ew.cummax() - 1).min() * 100)}},
        "curve": curve_out,
        "trades": sorted(trades, key=lambda t: t["entry"], reverse=True),
        "open": open_rows, "pending": pending,
        "skipped": skipped[-50:], "skipped_count": len(skipped),
        "signals_seen": int(len(ent)),
        "prices": px_out,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_clean(out), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    s = pf_stats
    print(f"[pf] {out['range'][0]} → {out['range'][1]} · พอร์ต {s['ret']}% (S&P {out['stats']['spx']['ret']}%, "
          f"EW {out['stats']['ew']['ret']}%) · max DD {s['max_dd']}% · {s['trades']} เทรดปิดแล้ว · "
          f"ถืออยู่ {len(open_rows)} · ข้ามเพราะพอร์ตเต็ม/เงินไม่พอ {len(skipped)}")
    if proxy_missing:
        print(f"[pf] ไม่มีราคา ETF ตัวแทน {sorted(set(proxy_missing))} — ใช้ราคาตัวต้นทางแทน")
    print(f"[ok] เขียน {OUT}")


if __name__ == "__main__":
    main()
