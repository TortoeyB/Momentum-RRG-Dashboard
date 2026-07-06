# Technical Score Rotation — RRG Dashboard

Dashboard สไตล์ RRG ที่เปลี่ยนแกนจาก "ราคาเทียบ SPY" เป็น **คะแนน technical รวม**
(Cipher B / Hull 9-20 / MA50 / MA200 + ADX เป็นตัวคูณ) พร้อม buy/sell signal
แบบ checklist, price structure และ candlestick pattern

- แกน X = คะแนนรวม 0–100 (เส้นแบ่ง 50) · แกน Y = Δ คะแนนย้อน 5/10 วัน
- Quadrant: Leading / Weakening / Lagging / Improving เหมือน RRG
- คะแนนธีม = ค่าเฉลี่ย equal-weight ของหุ้นสมาชิก · price action ใช้ ETF หลักของธีม

---

## เริ่มใช้ครั้งแรก (ทำครั้งเดียว)

1. สร้าง repo ใหม่บน GitHub (private ก็ได้) แล้ว push โฟลเดอร์นี้ทั้งหมดขึ้นไป
   ```bash
   git init && git add -A && git commit -m "init"
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
2. เปิด GitHub Pages: **Settings → Pages → Source = Deploy from a branch →
   Branch `main` / folder `/docs`** → รอ 1–2 นาที จะได้ URL dashboard
3. เปิดสิทธิ์ให้ Actions push ได้: **Settings → Actions → General →
   Workflow permissions → Read and write permissions**

จากนั้นระบบจะรันเองทุกวันจันทร์–ศุกร์ ~04:30 เช้าเวลาไทย (หลังตลาด US ปิด)
หรือกดรันเองได้ที่แท็บ **Actions → Update dashboard data → Run workflow**

## รันบนเครื่องตัวเอง

```bash
pip install -r requirements.txt
python scripts/run_daily.py            # ดึงข้อมูลจริง + สร้าง docs/data.json
python scripts/run_daily.py --demo     # ข้อมูลจำลอง (ทดสอบหน้าเว็บ)
cd docs && python -m http.server 8000  # เปิด http://localhost:8000
```

## เพิ่ม/แก้รายชื่อหุ้น

- **วิธีหลัก:** แก้ `themes.yaml` — เพิ่มธีม, ETF หรือหุ้นได้ตามโครงสร้างเดิม
- **วิธีเสริม (จาก TradingView):** เปิด watchlist → เมนู ⋯ → **Export list**
  ได้ไฟล์ .txt → เอามาวางชื่อ `watchlist.txt` ที่ root ของ repo
  ระบบจะ parse (ตัด prefix เช่น `NASDAQ:` ให้เอง) และดึงราคา symbol
  เหล่านั้นเพิ่มอัตโนมัติ

## Backup ขึ้น Google Drive

ทุกครั้งที่รัน ระบบเก็บสำเนา `backup/data_YYYY-MM-DD.json` ไว้ใน repo อยู่แล้ว
(ย้อนดูประวัติ signal ได้ผ่าน git) — ถ้าต้องการสำเนาบน Drive ด้วย เลือกได้สองทาง:

- **ง่ายสุด:** ติดตั้ง Google Drive for Desktop แล้ว clone repo ไว้ในโฟลเดอร์
  ที่ sync กับ Drive — โฟลเดอร์ `backup/` จะขึ้น Drive เองทุกครั้งที่ pull
- **อัตโนมัติเต็มรูปแบบ:** ใช้ `rclone` (`rclone copy backup/ gdrive:rrg-backup/`)
  ตั้งเป็น cron หรือเพิ่มเป็น step ใน GitHub Actions (ต้องตั้งค่า
  service account + secret)

## ปรับแต่งคะแนน / เกณฑ์ signal

- น้ำหนักคะแนน: `scripts/scoring.py` → `WEIGHTS` (ค่าเริ่มต้น Cipher 35 /
  Hull 25 / MA50 20 / MA200 20)
- เกณฑ์ ADX: ในฟังก์ชัน `score_series` (35 / 20 → ×1.25 / ×0.6)
- เกณฑ์ BUY/SELL (≥7) และ WATCH/REDUCE (4–6): ท้ายฟังก์ชัน `signal`
- โซน WaveTrend (±53, gold −75): `scripts/indicators.py`
- หน้าตา dashboard: แก้ `docs/index.html` ได้ตรงๆ (HTML/CSS/JS ไฟล์เดียว)

## โครงสร้าง

```
themes.yaml            รายชื่อธีม + ETF + หุ้น (ไฟล์หลักที่แก้บ่อยสุด)
watchlist.txt          (optional) export จาก TradingView
scripts/
  fetch_data.py        ดึง OHLCV (yfinance + cache) / parser watchlist / โหมด demo
  indicators.py        Cipher B (WaveTrend, MF, divergence, gold buy),
                       Hull, ADX, MA, candlestick patterns
  scoring.py           คะแนน 0–100 + quadrant + โครงสร้างราคา + signal checklist
  run_daily.py         pipeline หลัก → docs/data.json + backup/
docs/
  index.html           dashboard (GitHub Pages เสิร์ฟโฟลเดอร์นี้)
  data.json            ผลคะแนนล่าสุด
.github/workflows/update.yml   รันอัตโนมัติทุกวันทำการ
```

> ระบบนี้จัดระเบียบสัญญาณ technical เพื่อประกอบการตัดสินใจเท่านั้น
> ไม่ใช่คำแนะนำการลงทุน — ควร backtest และปรับเกณฑ์ให้เข้ากับสไตล์ของตัวเองก่อนใช้จริง
