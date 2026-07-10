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

import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_data import ROOT, load_themes, get_data, load_watchlists, load_names
from scoring import score_series, quadrant, price_structure, significant_pattern, signal

TAIL_LEN = 8        # จำนวนจุดของหางบนกราф RRG
HIST_LEN = 20       # เก็บคะแนนย้อนหลังกี่วัน (พอสำหรับ Δ10D + หาง 8 จุด)
PX_LEN = 11         # ราคาย้อนหลังสำหรับ sparkline / Δ1-10D


def pct_change(px: pd.Series, d: int) -> float:
    if len(px) <= d:
        return 0.0
    return round(float(px.iloc[-1] / px.iloc[-1 - d] - 1) * 100, 2)


def symbol_payload(sym: str, sc: pd.DataFrame, df: pd.DataFrame, name: str = "") -> dict:
    s = sc["score"]
    d5 = float(s.iloc[-1] - s.iloc[-6]) if len(s) > 5 else 0.0
    d5_prev = float(s.iloc[-6] - s.iloc[-11]) if len(s) > 10 else 0.0
    q_now = quadrant(float(s.iloc[-1]), d5)
    q_prev = quadrant(float(s.iloc[-6]), d5_prev) if len(s) > 10 else q_now
    struct = price_structure(df)
    sc2 = sc.copy()
    sc2["close"] = df["Close"]
    patt = significant_pattern(sc2)
    sig = signal(sc, q_now, q_prev, struct, patt)
    last = sc.iloc[-1]
    return {
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
        struct = price_structure(ref_df)
        sc2 = ref_sc.copy()
        sc2["close"] = ref_df["Close"]
        patt = significant_pattern(sc2)

        # signal ระดับธีม: quadrant จากคะแนนธีม, price action จาก ETF อ้างอิง,
        # สัญญาณ Cipher B จาก ETF อ้างอิง
        sig = signal(ref_sc, q_now, q_prev, struct, patt)

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
            "symbols": [symbol_payload(s, scores[s], data[s], names.get(s, ""))
                        for s in (t["etfs"] + t["stocks"]) if s in scores],
        })

    as_of = max(df.index[-1] for df in data.values()).strftime("%Y-%m-%d")
    watchlists_out = [{"name": k,
                       "symbols": [symbol_payload(x, scores[x], data[x], names.get(x, ""))
                                   for x in v if x in scores]}
                      for k, v in wl.items()]
    payload = {"as_of": as_of, "demo": demo, "tail_len": TAIL_LEN,
               "themes": themes_out, "watchlists": watchlists_out}

    out = os.path.join(ROOT, "docs", "data.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[done] เขียน {out}  ({len(themes_out)} ธีม, as of {as_of})")

    bdir = os.path.join(ROOT, "backup")
    os.makedirs(bdir, exist_ok=True)
    shutil.copy(out, os.path.join(bdir, f"data_{as_of}.json"))
    print(f"[done] backup → backup/data_{as_of}.json")


if __name__ == "__main__":
    main()
