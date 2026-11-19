"""
signals_log.py — บันทึกสัญญาณรายวันแบบ "ห้ามแก้ย้อนหลัง" สำหรับ performance tracker

ทำไมต้องมี (ทั้งที่มี backup/ อยู่แล้ว):
    backup/data_<as_of>.json ถูกเขียนทับทุกรอบที่ได้ as_of เดียวกัน
    (รอบหลัก 21:37 UTC + รอบสำรอง 01:17/05:43 UTC) → ไฟล์ = ผลของรอบสุดท้าย
    ไม่ใช่สัญญาณที่ "เห็นจริง" ตอนเช้า  log นี้เก็บเฉพาะ "รอบแรก" ของแต่ละ as_of
    และไม่แตะแถวเก่าอีกเลย

ไฟล์: data/signals_log.csv — 1 แถว = 1 symbol × 1 as_of
    as_of, sym, grade, bscore, score, quadrant, bar_date, close,
    watchlists, signal_version, generated_at, source

    grade    BUY / ACCUM / WATCH / HOLD / REDUCE / SELL / NONE (= ไม่มีสัญญาณ)
    bar_date วันที่ของแท่งที่ใช้คำนวณจริง (crypto/futures อาจช้ากว่า as_of 1 วัน)
    source   live   = บันทึกตอน pipeline รัน
             backup = เติมย้อนหลังจาก backup/ (ก่อนเริ่มมี log)
"""

import csv
import glob
import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "data", "signals_log.csv")
BACKUP_DIR = os.path.join(ROOT, "backup")

FIELDS = ["as_of", "sym", "grade", "bscore", "score", "quadrant", "bar_date",
          "close", "watchlists", "signal_version", "generated_at", "source"]


def version_for_date(as_of: str) -> str:
    """เวอร์ชันของ backup ที่เกิดก่อนมี signal_version ใน data.json (ดู scoring.SIGNAL_VERSION)"""
    if as_of < "2026-08-04":
        return "v1"
    return "v1.1"


def logged_dates(path: str = LOG_PATH) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {r["as_of"] for r in csv.DictReader(f)}


def _bar_date(as_of: str, label: str | None) -> str:
    """แปลง '18 Nov' ของ score_dates เป็น YYYY-MM-DD (ข้ามปีได้)"""
    if not label:
        return as_of
    y = int(as_of[:4])
    d = datetime.strptime(f"{label} {y}", "%d %b %Y")
    if d.strftime("%Y-%m-%d") > as_of and (d - datetime.strptime(as_of, "%Y-%m-%d")).days > 5:
        d = d.replace(year=y - 1)
    return d.strftime("%Y-%m-%d")


def rows_from_payload(payload: dict, source: str,
                      bar_dates: dict[str, str] | None = None) -> list[dict]:
    """แปลง data.json (หรือ backup) เป็นแถว log — 1 แถวต่อ symbol (รวมทุก watchlist)"""
    as_of = payload["as_of"]
    ver = payload.get("signal_version") or version_for_date(as_of)
    gen = payload.get("generated_at", "")
    rows: dict[str, dict] = {}
    for w in payload.get("watchlists", []):
        for s in w.get("symbols", []):
            sym = s["sym"]
            if sym in rows:
                rows[sym]["watchlists"] += "|" + w["name"]
                continue
            sig = s.get("signal") if isinstance(s.get("signal"), dict) else None
            px = s.get("px_hist") or []
            sd = s.get("score_dates") or []
            bd = (bar_dates or {}).get(sym) or _bar_date(as_of, sd[-1] if sd else None)
            rows[sym] = {
                "as_of": as_of, "sym": sym,
                "grade": sig["grade"] if sig else "NONE",
                "bscore": sig.get("score", "") if sig else "",
                "score": s.get("score", ""),
                "quadrant": s.get("quadrant", ""),
                "bar_date": bd,
                "close": px[-1] if px else "",
                "watchlists": w["name"],
                "signal_version": ver, "generated_at": gen, "source": source,
            }
    return list(rows.values())


def _append(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def append_live(payload: dict, bar_dates: dict[str, str] | None = None,
                path: str = LOG_PATH) -> int:
    """เรียกจาก run_daily หลังเขียน data.json — บันทึกเฉพาะ as_of ที่ยังไม่มี (รอบแรกชนะ)"""
    if payload.get("demo"):
        return 0
    if payload["as_of"] in logged_dates(path):
        print(f"[log] as_of={payload['as_of']} มีใน signals_log แล้ว — ไม่เขียนทับ (รอบแรกชนะ)")
        return 0
    rows = rows_from_payload(payload, "live", bar_dates)
    _append(rows, path)
    print(f"[log] บันทึกสัญญาณ {len(rows)} symbols ของ {payload['as_of']} ลง signals_log.csv")
    return len(rows)


def backfill_from_backups(path: str = LOG_PATH, backup_dir: str = BACKUP_DIR) -> int:
    """เติม as_of ที่ยังไม่มีใน log จาก backup/ (ใช้ครั้งแรก + กันวันที่ log หลุด)"""
    have = logged_dates(path)
    rows: list[dict] = []
    for f in sorted(glob.glob(os.path.join(backup_dir, "data_*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa
            print(f"[log] อ่าน {os.path.basename(f)} ไม่ได้ ({e}) — ข้าม")
            continue
        if not d.get("watchlists") or d.get("demo") or d["as_of"] in have:
            continue
        rows += rows_from_payload(d, "backup")
        have.add(d["as_of"])
    if rows:
        rows.sort(key=lambda r: (r["as_of"], r["sym"]))
        _append(rows, path)
        print(f"[log] เติมย้อนหลังจาก backup {len({r['as_of'] for r in rows})} วัน ({len(rows)} แถว)")
    return len(rows)
