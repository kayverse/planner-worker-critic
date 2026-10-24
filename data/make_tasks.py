"""Generates MOCK data + 20 auto-checkable tasks. Seeded, so results are reproducible."""
import csv, json, random, datetime as dt
from collections import defaultdict, Counter
random.seed(42)

REGIONS = ["North", "South", "East"]
PRODUCTS = {"Pen": 12, "Notebook": 45, "Bag": 300, "Lamp": 150, "Mug": 60}
N = 120

def month_name(d): return d.strftime("%B")

# ---------- clean dataset ----------
rows = []
for i in range(1, N + 1):
    d = dt.date(2026, 1, 1) + dt.timedelta(days=random.randint(0, 89))  # Jan-Mar
    p = random.choice(list(PRODUCTS))
    rows.append(dict(order_id=f"O{i:03d}", customer_id=f"C{random.randint(1,25):02d}",
        region=random.choice(REGIONS), product=p, units=random.randint(1, 6),
        price=PRODUCTS[p], date=d.isoformat(), refunded=random.random() < 0.1))
for r in rows: r["revenue"] = r["units"] * r["price"]

def write(path, rows_):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_[0].keys())); w.writeheader(); w.writerows(rows_)
write("sales_clean.csv", [{k: v for k, v in r.items() if k != "revenue"} for r in rows])

# ---------- dirty dataset (same orders, with realistic problems) ----------
dirty = [dict(r) for r in rows]
for r in dirty: r.pop("revenue")
random.shuffle(dirty)
# duplicates
dups = random.sample(dirty, 8); dirty += [dict(d) for d in dups]
# region case/whitespace
for r in dirty:
    x = random.random()
    if x < 0.2: r["region"] = r["region"].upper()
    elif x < 0.35: r["region"] = " " + r["region"].lower() + " "
# mixed date formats
for r in dirty:
    if random.random() < 0.3:
        y, m, d = r["date"].split("-"); r["date"] = f"{d}/{m}/{y}"
# missing prices
missing_ids = set(random.sample([r["order_id"] for r in rows], 6))
for r in dirty:
    if r["order_id"] in missing_ids: r["price"] = ""
write("sales_dirty.csv", dirty)

# ---------- ground truth helpers (from clean data) ----------
ok = [r for r in rows if not r["refunded"]]
def rev(rs): return sum(r["revenue"] for r in rs)
by_region_ok = defaultdict(int)
for r in ok: by_region_ok[r["region"]] += r["revenue"]
by_cust_ok = defaultdict(int)
for r in ok: by_cust_ok[r["customer_id"]] += r["revenue"]
by_month = defaultdict(int)
for r in rows: by_month[month_name(dt.date.fromisoformat(r["date"]))] += r["revenue"]
by_prod_rev = defaultdict(int); by_prod_units = defaultdict(int)
for r in rows: by_prod_rev[r["product"]] += r["revenue"]; by_prod_units[r["product"]] += r["units"]
top3 = sorted(by_prod_rev, key=by_prod_rev.get, reverse=True)[:3]
top_cust = max(by_cust_ok, key=by_cust_ok.get)
cust_months = defaultdict(set)
for r in rows: cust_months[r["customer_id"]].add(dt.date.fromisoformat(r["date"]).month)
both_jan_feb = sum(1 for m in cust_months.values() if {1, 2} <= m)
q1_orders = Counter(r["customer_id"] for r in rows)
top_orders_cust = max(q1_orders, key=lambda c: (q1_orders[c], c))
refund_counts = Counter(r["customer_id"] for r in rows if r["refunded"])
feb_orders = sum(1 for r in rows if dt.date.fromisoformat(r["date"]).month == 2)
valid_price_rows = [r for r in rows if r["order_id"] not in missing_ids]

T = []
def add(tier, q, ans, kind="number", data="sales_clean.csv", tol=0.01):
    T.append(dict(id=f"{tier}{sum(1 for t in T if t['tier']==tier)+1}", tier=tier, data_file=data,
                  question=q, expected=ans, kind=kind, tolerance=tol))

# Tier A: simple (1-2 steps) -> worker alone should mostly pass
add("A", "What is the total revenue (units x price) of all orders?", rev(rows))
add("A", "How many orders are in region North?", sum(r["region"]=="North" for r in rows))
add("A", "What is the average revenue per order, rounded to 2 decimals?", round(rev(rows)/len(rows), 2))
add("A", "Which product has the highest total units sold?", max(by_prod_units, key=by_prod_units.get), "text")
add("A", "How many distinct customers placed orders?", len({r["customer_id"] for r in rows}))

# Tier B: multi-step -> planner should help
add("B", "Revenue from region South in March only, excluding refunded orders?",
    rev([r for r in ok if r["region"]=="South" and dt.date.fromisoformat(r["date"]).month==3]))
add("B", "Which customer has the highest revenue excluding refunded orders, and how much? Answer as 'ID amount'.",
    f"{top_cust} {by_cust_ok[top_cust]}", "text")
add("B", "Which month has the highest total revenue (all orders)?", max(by_month, key=by_month.get), "text")
add("B", "Difference between the best and worst region by revenue excluding refunds?",
    max(by_region_ok.values()) - min(by_region_ok.values()))
add("B", "Average units per order across the top 3 products by total revenue, rounded to 2 decimals?",
    round(sum(r["units"] for r in rows if r["product"] in top3) / sum(1 for r in rows if r["product"] in top3), 2))
add("B", "What percent of orders were refunded, rounded to 1 decimal?",
    round(100*sum(r["refunded"] for r in rows)/len(rows), 1))
add("B", "How many customers ordered in both January and February?", both_jan_feb)
add("B", "Which customer placed the most orders overall (ties: highest ID), and how many of their orders were refunded? Answer as 'ID count'.",
    f"{top_orders_cust} {refund_counts.get(top_orders_cust,0)}", "text")

# Tier C: dirty data and traps -> critic should help
add("C", "Total revenue after removing duplicate order_ids?", rev(rows), data="sales_dirty.csv")
add("C", "How many orders per region? Ignore case and extra spaces. Answer as 'North South East' counts, e.g. '40 41 39'.",
    " ".join(str(sum(r["region"]==g for r in rows)) for g in REGIONS), "text", "sales_dirty.csv")
add("C", "Average price per unit across orders that have a price, rounded to 2 decimals (dedupe order_ids first)?",
    round(sum(r["price"] for r in valid_price_rows)/len(valid_price_rows), 2), data="sales_dirty.csv")
add("C", "How many distinct orders were placed in February? (dates are mixed YYYY-MM-DD and DD/MM/YYYY)",
    feb_orders, data="sales_dirty.csv")
add("C", "What is the total revenue for region West?", "NOT FOUND", "text", "sales_dirty.csv")
add("C", "Which region has the highest revenue excluding refunds? Dedupe, ignore case/spaces. (Rows with missing price: look the price up from the same product's other rows.)",
    max(by_region_ok, key=by_region_ok.get), "text", "sales_dirty.csv")
add("C", "How many distinct orders have a missing price?", len(missing_ids), data="sales_dirty.csv")

json.dump(T, open("tasks.json", "w"), indent=2)
print(len(T), "tasks;", Counter(t["tier"] for t in T))
for t in T: print(t["id"], "->", t["expected"])
