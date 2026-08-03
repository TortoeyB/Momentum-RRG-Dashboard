"""
run_daily.py — รันทีเดียวจบ: ดึงข้อมูล → คำนวณคะแนน → export JSON

ใช้งาน:
    python scripts/run_daily.py            # ข้อมูลจริงจาก yfinance
    python scripts/run_daily.py --demo     # ข้อมูลจำลอง (ทดสอบ)
    python scripts/run_daily.py --force    # ดึงราคาใหม่ทั้งหมด ไม่ใช้ cache

ผลลัพธ์:
    docs/data.json                         # dashboard อ่านไฟล์นี้
    backup/data_YYYY-MM-DD.json            # สำเนารายวัน (sync ขึ้น Drive ได้)
"""

import collections
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_data import ROOT, load_themes, get_data, load_watchlists, load_names
from data_quality import audit
from scoring import score_series, quadrant, price_structure, significant_pattern, signal, signal_with_age
import math


def _clean(o):
    """แปลง NaN/Inf -> null ก่อนเขียน JSON (browser parse NaN ไม่ได้)"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(x) for x in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o

TAIL_LEN = 8        # จำนวนจุดของหางบนกราф RRG
HIST_LEN = 20       # เก็บคะแนนย้อนหลังกี่วัน (พอสำหรับ Δ10D + หาง 8 จุด)
PX_LEN = 11         # ราคาย้อนหลังสำหรับ sparkline / Δ1-10D


def pct_change(px: pd.Series, d: int) -> float:
    if len(px) <= d:
        return 0.0
    return round(float(px.iloc[-1] / px.iloc[-1 - d] - 1) * 100, 2)


def demote(sig: dict | None, q: dict | None) -> dict | None:
    """ข้อมูลเชื่อไม่ได้ = ไม่มีสัญญาณ ไม่ใช่สัญญาณอ่อน

    กด grade ลง HOLD 0/10 เพื่อให้หลุดจากการ์ด "Setup เด่นวันนี้" (กรอง >=4)
    แต่ยังคง object ไว้ ไม่ลบทิ้ง — หน้าเว็บจะได้ขึ้นป้ายเตือนแทนช่องว่าง
    """
    if not sig or not q or q.get("status") != "bad":
        return sig
    return {"grade": "HOLD", "score": 0, "checklist": []}


def symbol_payload(sym: str, sc: pd.DataFrame, df: pd.DataFrame, name: str = "",
                   q: dict | None = None) -> dict:
    s = sc["score"]
    d5 = float(s.iloc[-1] - s.iloc[-6]) if len(s) > 5 else 0.0
    sig, struct, patt, q_now, q_prev = signal_with_age(sc, df, s)
    sig = demote(sig, q)
    last = sc.iloc[-1]
    return {
        **({"quality": q} if q else {}),
        "sym": sym,
        "name": name,
        "score_hist": [round(float(v), 1) for v in s.tail(HIST_LEN)],
        "score_dates": [d.strftime("%d %b") for d in s.tail(HIST_LEN).index],
        "px_hist": [round(float(v), 2) for v in df["Close"].tail(PX_LEN)],
        "chg": {f"d{d}": pct_change(df["Close"], d) for d in (1, 3, 5, 10)},
        "sub": {k: round(float(last[f"{k}_sc"]), 1)
                for k in ("cipher", "hull", "ma50", "ma200")},
        "adx_mult": round(float(last["adx_mult"]), 2),
        "score": round(float(s.iloc[-1]), 1),
        "delta5": round(d5, 1),
        "quadrant": q_now,
        "structure": struct,
        "pattern": patt,
        "signal": sig,
        "chg5": pct_change(df["Close"], 5),
    }


def main():
    demo = "--demo" in sys.argv
    force = "--force" in sys.argv
    cfg = load_themes()
    wl = load_watchlists(demo=demo)
    if demo and not wl:
        wl = {"Demo WL": ["NVDA", "MSFT", "GLD", "XLE", "COIN", "JPM", "LLY", "URA"]}
    extra = sorted({x for v in wl.values() for x in v})
    data = get_data(cfg, demo=demo, extra=extra, force=force)

    if not data:
        raise SystemExit("ไม่มีข้อมูลราคาเลย — ตรวจการเชื่อมต่อ/รายชื่อ symbol")

    # ตรวจ+ซ่อม split ที่ Yahoo ไม่ได้ adjust ก่อนคำนวณ indicator
    # ต้องอยู่ก่อน score_series เสมอ ไม่งั้น MA/HMA/WT พาดข้ามรอยต่อไปแล้ว
    data, quality = audit(data, demo=demo)

    names = load_names(list(data.keys()), demo=demo)

    print("[calc] คำนวณ indicator + คะแนนรายตัว ...")
    scores: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        try:
            scores[sym] = score_series(df)
        except Exception as e:  # noqa
            print(f"[warn] {sym}: คำนวณไม่สำเร็จ ({e})")

    themes_out = []
    for t in cfg["themes"]:
        members = [s for s in (t["stocks"] or t["etfs"]) if s in scores]
        if not members:
            print(f"[warn] ธีม {t['name']}: ไม่มีข้อมูลสมาชิก — ข้าม")
            continue

        # คะแนนธีม = ค่าเฉลี่ย equal-weight ของสมาชิก (align วันที่ร่วมกัน)
        panel = pd.concat({s: scores[s]["score"] for s in members}, axis=1).dropna()
        th_score = panel.mean(axis=1)
        if len(th_score) < HIST_LEN:
            print(f"[warn] ธีม {t['name']}: ประวัติสั้นเกิน — ข้าม")
            continue
        hist = [round(float(v), 1) for v in th_score.tail(HIST_LEN)]
        hist_dates = [d.strftime("%d %b") for d in th_score.tail(HIST_LEN).index]

        d5 = th_score.iloc[-1] - th_score.iloc[-6]
        d5_prev = th_score.iloc[-6] - th_score.iloc[-11]
        q_now = quadrant(float(th_score.iloc[-1]), float(d5))
        q_prev = quadrant(float(th_score.iloc[-6]), float(d5_prev))

        # sub-score เฉลี่ยของสมาชิก (โชว์ในตาราง)
        sub = {k: round(float(np.mean([scores[s][f"{k}_sc"].iloc[-1] for s in members])), 1)
               for k in ("cipher", "hull", "ma50", "ma200")}
        adx_mult = round(float(np.mean([scores[s]["adx_mult"].iloc[-1] for s in members])), 2)

        # price action ของธีม = ETF ตัวแรก (ถ้าไม่มีใช้สมาชิกตัวแรก)
        ref = next((e for e in t["etfs"] if e in data), members[0])
        ref_df, ref_sc = data[ref], scores[ref]
        # signal ระดับธีม: quadrant จากคะแนนธีม, price action/Cipher จาก ETF อ้างอิง
        sig, struct, patt, _, _ = signal_with_age(ref_sc, ref_df, th_score)
        sig = demote(sig, quality.get(ref))

        px = ref_df["Close"].tail(PX_LEN)
        themes_out.append({
            "name": t["name"],
            "group": t.get("group", "อื่นๆ"),
            "etf": ref,
            "members": members,
            "score_hist": hist,
            "score_dates": hist_dates,
            "sub": sub,
            "adx_mult": adx_mult,
            "px_hist": [round(float(v), 2) for v in px],
            "chg": {f"d{d}": pct_change(ref_df["Close"], d) for d in (1, 3, 5, 10)},
            "structure": struct,
            "pattern": patt,
            "signal": sig,
            "quadrant": q_now,
            **({"quality": quality[ref]} if ref in quality else {}),
            "symbols": [symbol_payload(s, scores[s], data[s], names.get(s, ""),
                                       quality.get(s))
                        for s in (t["etfs"] + t["stocks"]) if s in scores],
        })

    # as_of ต้องเป็นวันของ "ข้อมูลส่วนใหญ่" ไม่ใช่ max()
    # เพราะคริปโต/futures/FX/ดัชนีเอเชีย เดินวันเสาร์อาทิตย์และเร็วกว่าตลาด US
    # ทำให้ max() ลากวันที่ไปข้างหน้า หน้าจอจึงขึ้นวันที่ใหม่ทั้งที่ข้อมูลค้าง
    _lasts = [df.index[-1] for df in data.values() if len(df)]
    _mode = collections.Counter(_lasts).most_common(1)[0][0]
    as_of = _mode.strftime("%Y-%m-%d")
    _mx = max(_lasts)
    if _mx.normalize() > _mode.normalize():
        _ahead = sum(1 for x in _lasts if x.normalize() > _mode.normalize())
        print(f"[info] as_of={as_of} (จาก {sum(1 for x in _lasts if x.normalize()==_mode.normalize())} symbols) "
              f"· อีก {_ahead} ตัวถึง {_mx:%Y-%m-%d} แล้ว (คริปโต/FX/เอเชีย)")
    _lag = len(pd.bdate_range(_mode.normalize(), pd.Timestamp.today().normalize())) - 1
    if _lag >= 2:
        print(f"[WARN] ข้อมูลช้า {_lag} วันทำการ — แท่งล่าสุดคือ {as_of} "
              f"แต่วันนี้คือ {pd.Timestamp.today():%Y-%m-%d}")
    watchlists_out = []
    for k, v in wl.items():
        payload_syms = [symbol_payload(x, scores[x], data[x], names.get(x, ""),
                                       quality.get(x))
                        for x in v if x in scores]
        if payload_syms:
            watchlists_out.append({"name": k, "symbols": payload_syms})
        else:
            print(f"[warn] ลิสต์ {k}: ไม่มี symbol ที่ดึงข้อมูลได้เลย — ไม่สร้าง tab")
    n_bad = sum(1 for q in quality.values() if q.get("status") == "bad")
    payload = {"as_of": as_of, "demo": demo, "tail_len": TAIL_LEN,
               "quality_bad": n_bad, "quality_flagged": len(quality),
               "themes": themes_out, "watchlists": watchlists_out}

    out = os.path.join(ROOT, "docs", "data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, ensure_ascii=False, separators=(",", ":"))
    print(f"[ok] เขียน {out} — {len(themes_out)} ธีม, {len(watchlists_out)} watchlist")

    # สำเนารายวันไว้ใน backup/ (sync ขึ้น Drive ได้)
    backup_dir = os.path.join(ROOT, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, f"data_{as_of}.json")
    shutil.copyfile(out, backup)
    print(f"[ok] สำเนา {backup}")


if __name__ == "__main__":
    main()
