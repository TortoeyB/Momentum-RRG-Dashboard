"""
paper_portfolio.py — พอร์ตจำลองที่เทรดตามสัญญาณ dashboard → docs/portfolio.json

ใช้งาน:
    python scripts/paper_portfolio.py              # ราคาจาก yfinance (ใช้ cache ของ run_daily)
    python scripts/paper_portfolio.py --offline    # ราคาจาก px_hist ใน backup/ (ทดสอบ)

กติกา v2 (ใช้จริงตั้งแต่ 2026-11-20 · ช่วงก่อนหน้านั้นคือผลย้อนหลังที่รู้ผลแล้ว):
  เข้า   BUY = ไม้เต็ม (10% ของพอร์ต) · ACCUM = ครึ่งไม้ (5%)
         ถือ ACCUM อยู่แล้วเจอ BUY → เติมอีก 5% ให้เต็มไม้  (ทยอยซื้อ)
         เข้าเฉพาะวันแรกของ episode (ไม่ไล่ซื้อสัญญาณเดิมซ้ำ) · WATCH ไม่เข้า
  ออก   SELL → ขายหมด · REDUCE → ลด 1/3 ของไม้เต็มต่อ REDUCE รอบใหม่ (ทยอยขาย)
         ราคาปิดต่ำกว่าต้นทุนเฉลี่ย 8% → ขายหมด (stop loss) · ครบ HOLD_BARS แท่ง → ขายหมด
  ประวัติ  baseline-1 (REDUCE ขายหมด, ไม่มี stop) ใช้ 19 พ.ย. 2026 วันเดียว — เปลี่ยนเป็น v2 หลังทดสอบ
         v2 ถูกคิดขึ้นหลังเห็นข้อมูล ก.ค.–พ.ย. แล้ว ผลช่วงนั้นจึงดูดีเกินจริง ให้ดูผลหลัง LIVE_SINCE
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
REDUCE_FRAC = 1 / 3    # REDUCE ขายกี่ส่วนของไม้เต็ม (1.0 = ขายหมด · 1/3 = ลดทีละ 1/3 ต่อ REDUCE episode)
STOP_LOSS = 0.08       # ราคาปิดต่ำกว่าต้นทุนเฉลี่ย 8% → ขายหมดที่ราคาปิดนั้น · None = ไม่มี
LIVE_SINCE = "2026-11-20"   # วันแรกที่ใช้กติกานี้จริง — ผลก่อนวันนี้คือ backtest ที่รู้ผลแล้ว
COST = 0.0015
RULES_VERSION = "v2"

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
            p["units0"] = p["units"]
            p["adds"].append({"date": d.strftime("%Y-%m-%d"), "px": _r(price, 4), "grade": meta["grade"]})
        else:
            self.pos[inst] = {"units": units, "units0": units, "cost": value * (1 + COST), "entry": d, "entry_px": price,
                              "sym": meta["sym"], "grade": meta["grade"], "bars": 0,
                              "adds": [{"date": d.strftime("%Y-%m-%d"), "px": _r(price, 4), "grade": meta["grade"]}]}
        return True

    def sell(self, inst, d, reason, frac=1.0):
        """frac < 1 = ขายบางส่วน (สัดส่วนของไม้เต็ม units0) · บันทึกเป็นเทรดแยกตามสัดส่วนต้นทุน"""
        p = self.pos[inst]
        price = self.px(inst, d)
        units = p["units"] if frac >= 1 else min(p["units"], p["units0"] * frac)
        if units >= p["units"] * 0.999:
            units = p["units"]
        part = units / p["units"]
        proceeds = units * price * (1 - COST)
        self.cash += proceeds
        sold = {**p, "units": units, "cost": p["cost"] * part}
        self.trades.append(self._trade_row(inst, sold, d, price, proceeds, reason, open_=False))
        if part >= 1:
            self.pos.pop(inst)
        else:
            p["units"] -= units
            p["cost"] -= sold["cost"]

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
                    continue
            if STOP_LOSS and d in s.index and d > p["entry"]:
                avg = p["cost"] / p["units"] / (1 + COST)
                if s.loc[d] <= avg * (1 - STOP_LOSS):
                    book.sell(inst, d, f"stop −{STOP_LOSS * 100:.0f}%")
        # ---- 2) คำสั่งที่ถึงวันส่ง (ขายก่อน ซื้อทีหลัง) ----
        due = [o for o in pending if o["date"] == d]
        pending = [o for o in pending if o["date"] != d]
        for o in sorted(due, key=lambda o: o["side"] != "sell"):
            if o["side"] == "sell" and o["inst"] in book.pos:
                book.sell(o["inst"], d, o["reason"], o.get("frac", 1.0))
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
            prev_g, p["last_g"] = p.get("last_g"), g
            frac = 1.0
            if g == "REDUCE" and REDUCE_FRAC < 1:
                if prev_g == "REDUCE":
                    continue                    # ลดไปแล้วใน REDUCE รอบนี้ — รอ REDUCE รอบใหม่
                frac = REDUCE_FRAC
            if g in EXIT_GRADES and not any(o["inst"] == inst and o["side"] == "sell" for o in pending):
                # แท่งถัดไปยังไม่มีราคา (เช่นตลาดยังไม่ปิด) → เก็บเป็นคำสั่งรอส่ง date=None ไว้แสดงบนหน้าเว็บ
                # รอบถัดไปจำลองใหม่ทั้งหมดอยู่แล้ว จึงจะได้วันส่งจริงเมื่อมีราคาแท่งนั้น
                nb = next_bar(prices[inst], bar_of.get((p["sym"], d), d))
                pending.append({"date": nb, "side": "sell", "inst": inst, "frac": frac,
                                "reason": f"สัญญาณ {g}" + (" (ลด 1/3)" if frac < 1 else "")})
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
                nb = next_bar(prices[r.inst], r.entry_date)       # None = รอแท่งถัดไป (แสดงเป็นคำสั่งรอส่ง)
                w = W_BUY if r.grade == "BUY" else W_ACCUM
                if topup:
                    w = W_BUY - W_ACCUM
                pending.append({"date": nb, "side": "buy", "inst": r.inst, "weight": w, "topup": topup,
                                "meta": {"sym": r.sym, "grade": r.grade}})

    last = cal[-1]
    open_rows = []
    for inst, p in book.pos.items():
        price = book.px(inst, last)
        # unrealized = มูลค่าตามราคาปิดล่าสุด − ต้นทุน (ยังไม่หักค่าขาย เพราะยังไม่ได้ขาย)
        open_rows.append(book._trade_row(inst, p, last, price, p["units"] * price, "On Position", True))
    for inst, p in book.pos.items():                     # ถือครบกำหนดในแท่งถัดไป
        if p["bars"] == HOLD_BARS - 1 and not any(o["inst"] == inst and o["side"] == "sell" for o in pending):
            pending.append({"date": None, "side": "sell", "inst": inst, "reason": f"ครบ {HOLD_BARS} วัน"})
    pending_rows = [{"date": o["date"].strftime("%Y-%m-%d") if o["date"] is not None else None,
                     "side": o["side"], "inst": o["inst"],
                     **({"sym": o["meta"]["sym"], "grade": o["meta"]["grade"]} if "meta" in o else {"reason": o["reason"]})}
                    for o in pending]
    return pd.DataFrame(curve).set_index("d"), book.trades, open_rows, pending_rows, skipped, cal, ent, book.cash


def book_view(log: pd.DataFrame, prices: dict, srcs: list[str], live_from=None) -> dict:
    """จำลองพอร์ต 1 บัญชี แล้วคืนทุกอย่างที่หน้าเว็บต้องใช้ (curve, P/L, เทรด, สถิติ)"""
    curve, trades, open_rows, pending, skipped, cal, ent, cash = simulate(log, prices)
    if live_from is not None:
        ent = ent[ent.start >= pd.Timestamp(live_from)]
    pf = curve["pf"]
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
    return {
        "range": [cal[0].strftime("%Y-%m-%d"), cal[-1].strftime("%Y-%m-%d")],
        "pl": pl_block(pf, trades, open_rows, cash),
        "stats": {"portfolio": stats(pf, trades, curve["invested"]),
                  "spx": {"ret": _r((spx.iloc[-1] / spx.iloc[0] - 1) * 100),
                          "max_dd": _r((spx / spx.cummax() - 1).min() * 100)},
                  "ew": {"ret": _r((ew.iloc[-1] / ew.iloc[0] - 1) * 100),
                         "max_dd": _r((ew / ew.cummax() - 1).min() * 100)}},
        "curve": curve_out,
        "trades": sorted(trades, key=lambda t: t["entry"], reverse=True),
        "open": open_rows, "pending": pending,
        "skipped": skipped[-50:], "skipped_count": len(skipped),
        "signals_seen": int(len(ent)),
    }


def pl_block(pf: pd.Series, trades: list[dict], open_rows: list[dict], cash: float) -> dict:
    """สถานะพอร์ตปัจจุบัน: NAV = ทุน + realized + unrealized (ตรวจให้ลงตัว)"""
    nav = float(pf.iloc[-1])
    realized = float(sum(t["pnl"] or 0 for t in trades))
    unrealized = float(sum(t["pnl"] or 0 for t in open_rows))
    gap = nav - START_CAPITAL - realized - unrealized
    if abs(gap) > 50:                      # ปัดเศษรายเทรดได้ไม่กี่บาท — เกินนี้แปลว่าบัญชีไม่ลงตัว
        print(f"[pf] คำเตือน: NAV − ทุน ไม่เท่ากับ realized + unrealized (ต่าง {gap:,.0f})")
    prev = float(pf.iloc[-2]) if len(pf) > 1 else nav
    mkt = nav - cash
    return {"date": pf.index[-1], "capital": START_CAPITAL, "nav": _r(nav, 0),
            "total": _r(nav - START_CAPITAL, 0), "total_pct": _r((nav / START_CAPITAL - 1) * 100),
            "realized": _r(realized, 0), "realized_n": len(trades),
            "realized_win": sum(1 for t in trades if (t["pnl"] or 0) > 0),
            "unrealized": _r(unrealized, 0), "unrealized_n": len(open_rows),
            "unrealized_pct": _r(unrealized / (mkt - unrealized) * 100) if mkt - unrealized > 0 else None,
            "day": _r(nav - prev, 0), "day_pct": _r((nav / prev - 1) * 100) if prev else None,
            "prev_date": pf.index[-2] if len(pf) > 1 else None,
            "cash": _r(cash, 0), "cash_pct": _r(cash / nav * 100, 1), "invested": _r(mkt, 0)}


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

    # ---- 2 มุมมอง แยกบัญชีกันเด็ดขาด ----
    # backtest: ทั้งช่วงที่มีสัญญาณ (ก่อน LIVE_SINCE = ใช้กติกาย้อนหลัง → in-sample)
    # live:     เริ่มเงินสดเต็ม START_CAPITAL ใหม่ · เข้าเฉพาะสัญญาณที่เกิดตั้งแต่วันทำการก่อน LIVE_SINCE
    #           (สัญญาณวันนั้นรู้หลังตลาดปิด → ส่งคำสั่งแท่งแรกของ LIVE_SINCE) · ไม่รับ position ต่อจาก backtest
    bt = book_view(log, prices, srcs)
    snaps = sorted(log.as_of.unique())
    before = [d for d in snaps if d < pd.Timestamp(LIVE_SINCE)]
    live = None
    if len(before) >= 2:
        seed = before[-2]                     # แถวแรกของแต่ละตัวถูกใช้เป็น "เกรดก่อนหน้า" เท่านั้น
        live = book_view(log[log.as_of >= seed], prices, srcs, live_from=before[-1])
        base = before[-1].strftime("%Y-%m-%d")      # วันก่อนเริ่มจริง = จุดตั้งต้น 0% ของกราฟ
        live["curve"] = [c for c in live["curve"] if c["d"] >= base]
        live["range"][0] = base

    traded = sorted({t["inst"] for v in (bt, live) if v for t in v["trades"] + v["open"]})
    px_out = {}
    for inst in traded:
        s = prices[inst]
        s = s[s.index >= pd.Timestamp(bt["range"][0])]
        px_out[inst] = {"d": [x.strftime("%Y-%m-%d") for x in s.index], "c": [_r(v, 4) for v in s.values]}

    out = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "rules": {"version": RULES_VERSION, "capital": START_CAPITAL, "max_pos": MAX_POS,
                  "w_buy": W_BUY * 100, "w_accum": W_ACCUM * 100, "hold_bars": HOLD_BARS,
                  "exit_grades": sorted(EXIT_GRADES), "cost": COST * 100,
                  "reduce_frac": _r(REDUCE_FRAC, 4), "stop_loss": _r(STOP_LOSS * 100, 1) if STOP_LOSS else None,
                  "live_since": LIVE_SINCE,
                  "execution": "ราคาปิดแท่งถัดไปหลังเกิดสัญญาณ", "proxy": PROXY,
                  "proxy_missing": sorted(set(proxy_missing)),
                  "price_source": "backup px_hist" if offline else "yfinance (adjusted close)"},
        **bt,                                  # ระดับบนสุด = backtest (หน้าเว็บเดิมยังอ่านได้)
        "live": live,
        "prices": px_out,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(_clean(out), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    b = bt["stats"]["portfolio"]
    print(f"[pf] backtest {bt['range'][0]} → {bt['range'][1]} · {b['ret']}% (S&P {bt['stats']['spx']['ret']}%) · "
          f"DD {b['max_dd']}% · ขาย {b['trades']} · ถืออยู่ {len(bt['open'])}")
    if live:
        print(f"[pf] live ตั้งแต่ {LIVE_SINCE} · NAV {live['pl']['nav']:,.0f} · ขาย {len(live['trades'])} · "
              f"On Position {len(live['open'])} · คำสั่งรอส่ง {len(live['pending'])}")
    if proxy_missing:
        print(f"[pf] ไม่มีราคา ETF ตัวแทน {sorted(set(proxy_missing))} — ใช้ราคาตัวต้นทางแทน")
    print(f"[ok] เขียน {OUT}")


if __name__ == "__main__":
    main()
