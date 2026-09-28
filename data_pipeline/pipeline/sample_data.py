"""สร้างข้อมูลตัวอย่าง (สัดส่วนสถานะใกล้เคียงข้อมูลในแอป) พร้อมแถวเสียเล็กน้อยไว้ทดสอบ"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path


def generate(path, n=2000, seed=42, dirty_ratio=0.02, start=date(2026, 1, 1)):
    rng = random.Random(seed)
    statuses = rng.choices(["Completed", "Returned", "Cancelled", "Pending"],
                           weights=[82, 7, 6, 5], k=n)
    rows = []
    for i, status in enumerate(statuses, start=1):
        rows.append({
            "order_id": f"ORD{i:07d}",
            "customer_id": f"C{rng.randint(1, 500):04d}",
            "product_id": f"P{rng.randint(1, 60):03d}",
            "order_date": (start + timedelta(days=rng.randint(0, 180))).isoformat(),
            "order_status": status,
            "quantity": rng.randint(1, 5),
            "unit_price": round(rng.uniform(50, 2000), 2),
        })
    for row in rows:                                   # แถวสกปรก
        if rng.random() < dirty_ratio:
            kind = rng.choice(["blank_id", "neg_qty", "bad_date", "bad_status", "messy_status"])
            if kind == "blank_id":
                row["order_id"] = ""
            elif kind == "neg_qty":
                row["quantity"] = -1
            elif kind == "bad_date":
                row["order_date"] = "not-a-date"
            elif kind == "bad_status":
                row["order_status"] = "Shipped?"
            else:                                       # ยังแก้ได้ใน Silver
                row["order_status"] = f"  {row['order_status'].lower()} "
    rows += rng.sample(rows, k=max(1, n // 100))       # order_id ซ้ำ ~1%

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    print(generate(Path(__file__).resolve().parent.parent / "data" / "raw" / "orders_sample.csv"))
